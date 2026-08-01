// Tests for the llama-mmap advise bridge (MoE cache SSD/mmap tier).
//
// Runs the real llama_mmap against a temp file and asserts through
// /proc/self/smaps that DONTNEED drops mapped pages, that alignment is
// inward (a page shared with a neighbour is never dropped), and -- the
// safety contract -- that pointers outside a live mapping are never
// advised. The last one is what keeps MADV_DONTNEED away from anonymous
// memory, where it destroys data instead of dropping clean pages.
//
// Everything here is Linux-specific (smaps, MADV_DONTNEED semantics);
// on other platforms the test reports itself skipped.

#include "llama-mmap.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;
static std::string g_case;

#define CASE(name) do { g_case = (name); } while (0)

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL [%s] %s:%d: %s\n",                            \
                g_case.c_str(), __FILE__, __LINE__, #cond);                 \
        g_failures++;                                                       \
    }                                                                       \
} while (0)

#if defined(__linux__)

#include <unistd.h>
#include <sys/mman.h>

static size_t PAGE = 0;

// Sum of Rss over every VMA intersecting [addr, addr+len), from
// /proc/self/smaps. mincore() is the obvious oracle but reports page
// CACHE residency for file mappings, which MADV_DONTNEED does not
// change; what DONTNEED drops is this process's PTEs, and that is
// exactly what smaps Rss counts.
static long rss_kb(const void * addr, size_t len) {
    FILE * f = fopen("/proc/self/smaps", "r");
    if (!f) return -1;
    const uintptr_t lo = (uintptr_t) addr;
    const uintptr_t hi = lo + len;
    char line[512];
    bool in_range = false;
    long total = 0;
    while (fgets(line, sizeof(line), f)) {
        uintptr_t start = 0;
        uintptr_t end = 0;
        if (sscanf(line, "%lx-%lx ", &start, &end) == 2) {
            in_range = start < hi && end > lo;
        } else if (in_range && strncmp(line, "Rss:", 4) == 0) {
            long kb = 0;
            if (sscanf(line + 4, "%ld", &kb) == 1) total += kb;
        }
    }
    fclose(f);
    return total;
}

// Fault every page of the range into this process (read-only touch).
static void touch(const void * addr, size_t len) {
    volatile const uint8_t * p = (const uint8_t *) addr;
    volatile uint8_t sink = 0;
    for (size_t off = 0; off < len; off += PAGE) {
        sink ^= p[off];
    }
    (void) sink;
}

static std::string make_temp_model(size_t bytes) {
    const char * tmpdir = getenv("TMPDIR");
    std::string templ = std::string(tmpdir ? tmpdir : "/tmp") + "/test-mmap-advise-XXXXXX";
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    int fd = mkstemp(buf.data());
    if (fd < 0) {
        return "";
    }
    std::vector<uint8_t> pattern(bytes);
    for (size_t i = 0; i < bytes; i++) pattern[i] = (uint8_t)(i * 131 + 7);
    const ssize_t written = write(fd, pattern.data(), bytes);
    close(fd);
    if (written != (ssize_t) bytes) return "";
    return std::string(buf.data());
}

int main() {
    PAGE = (size_t) sysconf(_SC_PAGESIZE);
    // 6 full pages plus an odd tail, so the mapping end is not
    // page-aligned -- the case the inward rounding must survive.
    const size_t file_bytes = 6 * PAGE + 100;
    const std::string path = make_temp_model(file_bytes);
    if (path.empty()) {
        fprintf(stderr, "SKIP: could not create temp file\n");
        return 0;
    }

    {
        CASE("mapping registers the advise bridge");
        llama_file file(path.c_str(), "rb");
        // prefetch=0: no MAP_POPULATE, pages start unmapped in this
        // process so the Rss assertions below start from zero.
        llama_mmap mapping(&file, /*prefetch=*/0, /*numa=*/false);
        uint8_t * base = (uint8_t *) mapping.addr();

        ggml_backend_moe_mmap_advise_fn advise = ggml_backend_moe_get_mmap_advise_fn();
        CHECK(advise != nullptr);
        if (!advise) {
            fprintf(stderr, "no bridge registered; aborting remaining cases\n");
            return 1;
        }

        CASE("DONTNEED drops exactly the fully-covered pages");
        touch(base, file_bytes);
        const long before = rss_kb(base, file_bytes);
        CHECK(before >= (long)(6 * PAGE / 1024));

        advise(base + PAGE, 3 * PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
        const long after = rss_kb(base, file_bytes);
        CHECK(before - after == (long)(3 * PAGE / 1024));

        CASE("DONTNEED rounds inward, never dropping a shared page");
        touch(base, file_bytes);
        const long full = rss_kb(base, file_bytes);
        // Interior range covering no complete page: must be a no-op.
        advise(base + PAGE + 100, PAGE - 200, GGML_MOE_MMAP_ADVISE_DONTNEED);
        CHECK(rss_kb(base, file_bytes) == full);
        // Misaligned range [P+100, 3P+100) fully covers only page 2:
        // exactly one page drops.
        advise(base + PAGE + 100, 2 * PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
        CHECK(full - rss_kb(base, file_bytes) == (long)(PAGE / 1024));

        CASE("WILLNEED accepts a misaligned range");
        advise(base + PAGE + 100, 2 * PAGE, GGML_MOE_MMAP_ADVISE_WILLNEED);
        // Advisory readahead has no deterministic observable here; the
        // check is that a misaligned start is accepted (posix_madvise
        // needs an aligned address, so a pass-through would EINVAL and,
        // more importantly, a crash/UB would fail the run).

        CASE("a range not wholly inside the mapping is refused");
        touch(base + 4 * PAGE, 2 * PAGE);
        const long tail_before = rss_kb(base + 4 * PAGE, 2 * PAGE);
        // Extends past the mapping end: refused whole, nothing dropped.
        advise(base + 4 * PAGE, 3 * PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
        CHECK(rss_kb(base + 4 * PAGE, 2 * PAGE) == tail_before);

        CASE("DONTNEED never touches anonymous memory");
        void * anon = nullptr;
        CHECK(posix_memalign(&anon, PAGE, 3 * PAGE) == 0);
        memset(anon, 0xAB, 3 * PAGE);
        advise(anon, 3 * PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
        bool intact = true;
        for (size_t i = 0; i < 3 * PAGE; i++) {
            if (((uint8_t *) anon)[i] != 0xAB) { intact = false; break; }
        }
        // If the registry had let this through, MADV_DONTNEED would have
        // zeroed the pages.
        CHECK(intact);
        free(anon);

        CASE("advice inside an unmapped fragment is refused");
        mapping.unmap_fragment(PAGE, 2 * PAGE);
        // The hole: containment fails, no syscall, no crash.
        advise(base + PAGE, PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
        advise(base + PAGE - 100, PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
        // A surviving fragment still works.
        touch(base + 2 * PAGE, 2 * PAGE);
        const long frag_before = rss_kb(base + 2 * PAGE, 2 * PAGE);
        advise(base + 2 * PAGE, 2 * PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
        CHECK(frag_before - rss_kb(base + 2 * PAGE, 2 * PAGE) == (long)(2 * PAGE / 1024));
    }

    {
        CASE("a destroyed mapping's addresses are refused");
        // The mapping above is gone; its registry entries must be too.
        // There is no valid pointer to test against (the VMA is unmapped),
        // so this asserts on a fresh anonymous block at whatever address
        // the allocator gives us: still refused, still intact.
        ggml_backend_moe_mmap_advise_fn advise = ggml_backend_moe_get_mmap_advise_fn();
        CHECK(advise != nullptr);
        if (advise) {
            void * anon = nullptr;
            CHECK(posix_memalign(&anon, PAGE, PAGE) == 0);
            memset(anon, 0x5C, PAGE);
            advise(anon, PAGE, GGML_MOE_MMAP_ADVISE_DONTNEED);
            CHECK(((uint8_t *) anon)[0] == 0x5C && ((uint8_t *) anon)[PAGE - 1] == 0x5C);
            free(anon);
        }
    }

    unlink(path.c_str());

    if (g_failures) {
        fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all mmap advise tests passed\n");
    return 0;
}

#else // !__linux__

int main() {
    printf("mmap advise tests skipped (Linux-only oracle)\n");
    return 0;
}

#endif

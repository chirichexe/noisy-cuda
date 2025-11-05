/*
 * options.c - implementation of command-line parsing for Perlin generator.
 *
 */

/*
 * Copyright 2025 Davide Chirichella, Filippo Giulietti
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define _POSIX_C_SOURCE 200809L

#include "options.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctype.h>

/* Allowed formats - canonical lowercase list */
static const char *allowed_formats[] = { "png", "raw", "csv", "ppm", NULL };

/* forward declarations for helpers */
static void print_usage(FILE *out, const char *progname);
static void to_lower_inplace(char *s);
static int valid_format(const char *fmt);
static int parse_size_str(const char *s, int *w, int *h);
static int parse_unsigned_ull(const char *s, unsigned long long *out);
static int default_output_name(const char *fmt, char *outbuf, size_t buflen);
static uint64_t generate_seed_fallback(void);
static int try_read_urandom(uint64_t *seed_out);

/* print_usage
 *
 * Strict help following GNU conventions.
 */
static void print_usage(FILE *out, const char *progname) {
    fprintf(out,
        "Usage: %s [OPTIONS] [seed]\n"
        "\n"
        "Perlin noise generator — option parsing module only.\n"
        "\n"
        "  -h, --help                Show this help message and exit\n"
        "  -s, --size <WxH>          Image size in pixels (width x height). Default: 2048x2048\n"
        "  -O, --octaves <int>       Number of octaves (>=1). Default: 1\n"
        "  -f, --format <string>     Output format: png, raw, csv, ppm. Default: png\n"
        "  -o, --output <filename>   Output filename. Default: perlin.<ext>\n"
        "  -v, --verbose             Print processing steps and timings\n"
        "      --cpu                 Force CPU-only execution (no CUDA)\n"
        "  -S, --seed <uint64>       Provide explicit seed (alternative to positional)\n"
        "\n"
        "Positional 'seed' is accepted as an unsigned integer: if present it is\n"
        "interpreted as the RNG seed (e.g. './perlin 13813'). If both positional\n"
        "seed and --seed are provided the parser fails (ambiguous).\n"
        "\n",
        progname);
}



/* lowercase inplace */
static void to_lower_inplace(char *s) {
    for (; *s; ++s) *s = (char)tolower((unsigned char)*s);
}

/* check format */
static int valid_format(const char *fmt) {
    if (!fmt) return 0;
    char buf[32];
    strncpy(buf, fmt, sizeof(buf)-1);
    buf[sizeof(buf)-1] = '\0';
    to_lower_inplace(buf);
    for (const char **p = allowed_formats; *p; ++p) {
        if (strcmp(buf, *p) == 0) return 1;
    }
    return 0;
}

/* parse_positive_int style for WIDTHxHEIGHT */
static int parse_size_str(const char *s, int *w, int *h) {
    if (!s || !w || !h) return 0;
    const char *sep = strchr(s, 'x');
    if (!sep) sep = strchr(s, 'X');
    if (!sep) return 0;

    size_t left_len = (size_t)(sep - s);
    size_t right_len = strlen(sep + 1);
    if (left_len == 0 || right_len == 0) return 0;
    if (left_len > 20 || right_len > 20) return 0;

    char left[32], right[32];
    if (left_len >= sizeof(left) || right_len >= sizeof(right)) return 0;
    memcpy(left, s, left_len); left[left_len] = '\0';
    strncpy(right, sep + 1, sizeof(right)-1); right[sizeof(right)-1] = '\0';

    /* trim spaces */
    char *l = left; while (*l && isspace((unsigned char)*l)) ++l;
    char *lend = left + strlen(left) - 1; while (lend > l && isspace((unsigned char)*lend)) *lend-- = '\0';
    char *r = right; while (*r && isspace((unsigned char)*r)) ++r;
    char *rend = right + strlen(right) - 1; while (rend > r && isspace((unsigned char)*rend)) *rend-- = '\0';

    errno = 0;
    char *endptr = NULL;
    long wl = strtol(l, &endptr, 10);
    if (errno || endptr == l || *endptr != '\0' || wl <= 0 || wl > INT_MAX) return 0;
    errno = 0;
    long hl = strtol(r, &endptr, 10);
    if (errno || endptr == r || *endptr != '\0' || hl <= 0 || hl > INT_MAX) return 0;

    *w = (int)wl;
    *h = (int)hl;
    return 1;
}

/* parse unsigned long long strictly */
static int parse_unsigned_ull(const char *s, unsigned long long *out) {
    if (!s || !out) return 0;
    errno = 0;
    char *endptr = NULL;
    unsigned long long val = strtoull(s, &endptr, 10);
    if (errno != 0) return 0;
    if (endptr == s || *endptr != '\0') return 0;
    *out = val;
    return 1;
}

/* default output filename builder */
static int default_output_name(const char *fmt, char *outbuf, size_t buflen) {
    if (!fmt || !outbuf) return 0;
    int r = snprintf(outbuf, buflen, "perlin.%s", fmt);
    return (r > 0 && (size_t)r < buflen) ? 1 : 0;
}

/* try_read_urandom: attempt to fill seed_out from /dev/urandom */
static int try_read_urandom(uint64_t *seed_out) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return 0;
    uint64_t val = 0;
    ssize_t r = read(fd, &val, sizeof(val));
    close(fd);
    if (r != (ssize_t)sizeof(val)) return 0;
    *seed_out = val;
    return 1;
}

/* generate_seed_fallback: time + pid fallback */
static uint64_t generate_seed_fallback(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    pid_t pid = getpid();
    uint64_t v = ((uint64_t)ts.tv_nsec) ^ ((uint64_t)ts.tv_sec << 32) ^ ((uint64_t)pid << 16);
    /* mix a little */
    v ^= (v << 13);
    v ^= (v >> 7);
    v ^= (v << 17);
    return v;
}

/*
 * parse_program_options
 *
 * Implementation that does not exit. Caller decides exit codes.
 */
int parse_program_options(int argc, char **argv, ProgramOptions *out) {
    if (!out) {
        fprintf(stderr, "Internal error: NULL out parameter to parse_program_options\n");
        return -1;
    }

    /* defaults */
    out->width = 2048;
    out->height = 2048;

    out->octaves = 1;

    strncpy(out->format, "png", sizeof(out->format)-1);
    
    out->format[sizeof(out->format)-1] = '\0';
    out->output_filename[0] = '\0';
    out->verbose = 0;
    out->cpu_mode = 0;
    out->seed = 0;
    out->seed_provided = 0;

    const char *short_opts = "hs:O:f:o:vS:";
    const struct option long_opts[] = {
        {"help",    no_argument,       0, 'h'},
        {"size",    required_argument, 0, 's'},
        {"octaves", required_argument, 0, 'O'},
        {"format",  required_argument, 0, 'f'},
        {"output",  required_argument, 0, 'o'},
        {"verbose", no_argument,       0, 'v'},
        {"seed",    required_argument, 0, 'S'},
        {"cpu",     no_argument,       0, 1000},
        {0,0,0,0}
    };

    int opt;
    int opt_index = 0;
    optind = 1; /* reset getopt external state */

    /* track if seed provided via --seed */
    int seed_given_via_option = 0;
    unsigned long long parsed_seed_opt = 0;

    while ((opt = getopt_long(argc, argv, short_opts, long_opts, &opt_index)) != -1) {
        switch (opt) {
            case 'h':
                print_usage(stdout, argv[0]);
                return 1; /* help printed */

            case 's': {
                int w, h;
                if (!parse_size_str(optarg, &w, &h)) {
                    fprintf(stderr, "Error: invalid --size '%s'. Expected WIDTHxHEIGHT positive integers.\n", optarg);
                    return -1;
                }
                out->width = w;
                out->height = h;
                break;
            }

            case 'O': {
                long v;
                errno = 0;
                char *endptr = NULL;
                v = strtol(optarg, &endptr, 10);
                if (errno || endptr == optarg || *endptr != '\0' || v < 1 || v > 1000) {
                    fprintf(stderr, "Error: invalid --octaves '%s'. Must be integer >=1 (upper bound 1000).\n", optarg);
                    return -1;
                }
                out->octaves = (int)v;
                break;
            }

            case 'f':
                if (!valid_format(optarg)) {
                    fprintf(stderr, "Error: unsupported --format '%s'. Supported: png, raw, csv, ppm.\n", optarg);
                    return -1;
                }
                strncpy(out->format, optarg, sizeof(out->format)-1);
                out->format[sizeof(out->format)-1] = '\0';
                to_lower_inplace(out->format);
                break;

            case 'o':
                strncpy(out->output_filename, optarg, sizeof(out->output_filename)-1);
                out->output_filename[sizeof(out->output_filename)-1] = '\0';
                break;

            case 'v':
                out->verbose = 1;
                break;

            case 'S':
                if (!parse_unsigned_ull(optarg, &parsed_seed_opt)) {
                    fprintf(stderr, "Error: invalid --seed '%s'. Expect unsigned integer.\n", optarg);
                    return -1;
                }
                seed_given_via_option = 1;
                break;

            case 1000: /* --cpu */
                out->cpu_mode = 1;
                break;

            case '?':
            default:
                fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
                return -1;
        }
    }

    /* After options, remaining non-option args may contain positional seed */
    int positional_seed_provided = 0;
    unsigned long long parsed_seed_pos = 0;
    if (optind < argc) {
        /* Only accept a single positional argument (the seed). Any more -> error. */
        if (optind + 1 < argc) {
            fprintf(stderr, "Error: unexpected extra positional arguments. Only optional 'seed' is accepted.\n");
            return -1;
        }
        if (!parse_unsigned_ull(argv[optind], &parsed_seed_pos)) {
            fprintf(stderr, "Error: invalid positional seed '%s'. Expect unsigned integer.\n", argv[optind]);
            return -1;
        }
        positional_seed_provided = 1;
    }

    /* Ambiguity check: both provided -> error (strict) */
    if (seed_given_via_option && positional_seed_provided) {
        fprintf(stderr, "Error: seed provided both as positional argument and with --seed. Ambiguous.\n");
        return -1;
    }

    if (seed_given_via_option) {
        out->seed = (uint64_t)parsed_seed_opt;
        out->seed_provided = 1;
    } else if (positional_seed_provided) {
        out->seed = (uint64_t)parsed_seed_pos;
        out->seed_provided = 1;
    } else {
        /* generate seed: prefer /dev/urandom */
        uint64_t s = 0;
        if (try_read_urandom(&s)) {
            out->seed = s;
            out->seed_provided = 0;
        } else {
            out->seed = generate_seed_fallback();
            out->seed_provided = 0;
        }
    }

    /* Ensure output filename is set */
    if (out->output_filename[0] == '\0') {
        if (!default_output_name(out->format, out->output_filename, sizeof(out->output_filename))) {
            fprintf(stderr, "Fatal: cannot construct default output filename.\n");
            return -1;
        }
    }

    /* final sanity checks */
    if (out->width <= 0 || out->height <= 0) {
        fprintf(stderr, "Error: invalid image size (%d x %d).\n", out->width, out->height);
        return -1;
    }
    if (out->octaves < 1) {
        fprintf(stderr, "Error: octaves must be >= 1 (given %d).\n", out->octaves);
        return -1;
    }

    /* all good */
    return 0;
}

void print_program_options(ProgramOptions opts) {
    if (opts.verbose) {
        printf("\n");
        printf("              .__                                         .___       \n");
        printf("  ____   ____ |__| _________.__.           ____  __ __  __| _/____   \n");
        printf(" /    \\ /  _ \\|  |/  ___<   |  |  ______ _/ ___\\|  |  \\/ __ |\\__  \\  \n");
        printf("|   |  (  <_> )  |\\___ \\ \\___  | /_____/ \\  \\___|  |  / /_/ | / __ \\_\n");
        printf("|___|  /\\____/|__/____  >/ ____|          \\___  >____/\\____ |(____  /\n");
        printf("     \\/               \\/ \\/                   \\/           \\/     \\/ \n");
        printf("\n");

        fprintf(stderr, "Configuration (strict):\n");
        fprintf(stderr, "  Size:        %d x %d\n", opts.width, opts.height);
        fprintf(stderr, "  Octaves:     %d\n", opts.octaves);
        fprintf(stderr, "  Format:      %s\n", opts.format);
        fprintf(stderr, "  Output file: %s\n", opts.output_filename);
        fprintf(stderr, "  Backend:     %s\n", opts.cpu_mode ? "CPU (forced)" : "CUDA (default)");
        if (opts.seed_provided) {
            fprintf(stderr, "  Seed:        %" PRIu64 " (provided)\n", opts.seed);
        } else {
            fprintf(stderr, "  Seed:        %" PRIu64 " (auto-generated)\n", opts.seed);
        }
        fprintf(stderr, "  Verbose:     enabled\n");
    }
}
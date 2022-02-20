#include<utils/Palette.h>


Palette defaultPalette()
{
  // Source: https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia
  return {
    24, 82, 177,  // blue 2
    57, 125, 209,  // blue 1
    134, 181, 229,  // blue 0
    211, 236, 248,  // lightest blue
    241, 233, 191,  // lightest yellow
    248, 201, 95,  // light yellow
    255, 170, 0,  // dirty yellow
    204, 128, 0,  // brown 0
    153, 87, 0,  // brown 1
    106, 52, 3,  // brown 2
    66, 30, 15,  // brown 3
    25, 7, 26,  // dark violett

    47, 1, 9,
    73, 4, 4,
    100, 7, 0,
    138, 26, 12,
    177, 32, 24,
    209, 96, 57,
    229, 150, 134,
    248, 238, 211,

    181, 229, 134,
    125, 209, 57,
    82, 177, 24,
    44, 138, 12,
    7, 100, 0,
    4, 73, 4,
    1, 47, 9,

    9, 1, 47,  // darkest blue
    4, 4, 73,  // blue 5
    0, 7, 100,  // blue 4
    12, 44, 138,  // blue 3
  };
}
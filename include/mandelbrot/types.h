#pragma once

enum Device {
  CPU = 1,
  GPU = 2
};


struct Point {
  long double x, y;
};

struct Viewport {
  long double x1, x2, y1, y2;
};
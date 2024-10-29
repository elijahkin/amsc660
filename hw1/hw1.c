#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

bool is_nonassociative(float x, float y, float z) {
  float lhs = (x + y) + z;
  float rhs = x + (y + z);
  printf("lhs=%.8f\nrhs=%.8f\n", lhs, rhs);
  return lhs != rhs;
}

bool is_nondistributive(float x, float y, float z) {
  float lhs = x * z + y * z;
  float rhs = (x + y) * z;
  printf("lhs=%.8f\nrhs=%.8f\n", lhs, rhs);
  return lhs != rhs;
}

int main() {
  // Addition isn't associative
  assert(is_nonassociative(1.0, 0.0000002, 0.0000003));

  // Multiplication isn't distributive
  const float oneThird = 1 / (float)3;
  assert(is_nondistributive(1.000003, 0.000002, oneThird));
  return 0;
}

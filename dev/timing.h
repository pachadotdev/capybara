// #pragma once
// #include <chrono>
// #include <iostream>
// struct ScopedTimer {
//   const char* label;
//   std::chrono::high_resolution_clock::time_point start;
//   ScopedTimer(const char* l) : label(l), start(std::chrono::high_resolution_clock::now()) {}
//   ~ScopedTimer() {
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cerr << "[C++ timing] " << label << ": "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
//               << " ms" << std::endl;
//   }
// };
// #define TIME_FUNCTION ScopedTimer timer(__FUNCTION__);

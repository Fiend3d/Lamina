#pragma once

#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace lamina::model::prefix_io {
inline void write(std::ostream& out, const void* data, size_t bytes) {
    out.write(static_cast<const char*>(data), std::streamsize(bytes));
    if (!out) throw std::runtime_error("prefix snapshot write failed");
}
inline void read(std::istream& in, void* data, size_t bytes) {
    in.read(static_cast<char*>(data), std::streamsize(bytes));
    if (!in) throw std::runtime_error("prefix snapshot truncated");
}
template<class T> void put(std::ostream& out, const T& value) { write(out, &value, sizeof(value)); }
template<class T> T get(std::istream& in) { T value{}; read(in, &value, sizeof(value)); return value; }
template<class T> void put_vector(std::ostream& out, const std::vector<T>& values) {
    put(out, uint64_t(values.size())); write(out, values.data(), values.size() * sizeof(T));
}
template<class T> void get_vector(std::istream& in, std::vector<T>& values, size_t limit) {
    const auto size = get<uint64_t>(in);
    if (size > limit) throw std::runtime_error("prefix snapshot vector exceeds limit");
    values.resize(size); read(in, values.data(), values.size() * sizeof(T));
}
} // namespace lamina::model::prefix_io

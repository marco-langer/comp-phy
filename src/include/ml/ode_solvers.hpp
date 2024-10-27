#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>
#include <vector>


namespace ml::ode {

template <typename T, typename U>
concept BinaryFunc = std::is_same_v<std::invoke_result_t<T, U, U>, U>;


template <typename It, std::floating_point F, BinaryFunc<F> Func>
requires std::output_iterator<It, F> void euler(It out, std::size_t n, F h, F t_0, F y_0, Func func)
{
    F y_n = y_0;
    *out = y_n;
    for (std::size_t i = 0; i < n; ++i) {
        const F t = i * h + t_0;
        const F y_n2 = y_n + h * func(t, y_n);
        *out = y_n2;
        y_n = y_n2;
    }
}


template <std::floating_point F, BinaryFunc<F> Func>
void euler(std::vector<F>& data, std::size_t n, F h, F t_0, F y_0, Func func)
{
    data.clear();
    data.reserve(n + 1);
    euler(std::back_inserter(data), n, h, t_0, y_0, std::move(func));
}


template <std::floating_point F, BinaryFunc<F> Func>
std::vector<F> euler(std::size_t n, F h, F t_0, F y_0, Func func)
{
    std::vector<F> data;
    euler(data, n, h, t_0, y_0, std::move(func));
    return data;
}

}   // namespace ml::ode

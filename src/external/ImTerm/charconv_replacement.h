/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_EXTERNAL_IMTERM_CHARCONV_REPLACEMENT_H_
#define TRAACT_GUI_SRC_EXTERNAL_IMTERM_CHARCONV_REPLACEMENT_H_

#if __has_include(<charconv>)
#include <charconv>
#else
#include <string_view>
#include <system_error>

namespace std {
struct from_chars_result {
    const char* ptr;
    std::errc ec;
};

template<typename T>
from_chars_result from_chars( const char* first, const char* last, T& value,
                                   int fmt )  {
    char* p_end{nullptr};
    value = std::strtol(first, &p_end, fmt);
    if(first == p_end) {
        return {p_end, errc::invalid_argument};
    } else {
        return {p_end, errc()};
    }

    return from_chars_result();
};

}

#endif

#endif //TRAACT_GUI_SRC_EXTERNAL_IMTERM_CHARCONV_REPLACEMENT_H_

#ifndef __MAKEMORE_CMDLINE_HH__
#define __MAKEMORE_CMDLINE_HH__ 1

#include <string>
#include <map>
#include <set>

#include "youtil.hh"

namespace makemore {

struct Cmdline {
  struct Value {
    std::string x;

    Value() { }
    Value(const std::string &_x) : x(_x) { }
    Value(const char *_x) : x(_x) { }

    operator const std::string &() const
      { return x; }
    operator const char *() const
      { return x.c_str(); }
    operator int() const
      { return strtoi(x); }
    operator double() const
      { return strtod(x); }
  };

  std::string cmd;
  std::map<std::string, Value> m;
  std::set<std::string> unused;

  Cmdline(int *argcp, char ***argvp, const char *dopt);

  void put(const std::string &opt, const std::string &val);
  const Value &get(const std::string &opt);
  const Value &get(const std::string &opt, const std::string &dval);
  const Value &operator [](const std::string &opt);
};

}
#endif
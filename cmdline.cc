#define __MAKEMORE_CMDLINE_CC__ 1

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <string>

#include "cmdline.hh"
#include "youtil.hh"

namespace makemore {

static void parsecmd(int *argcp, char ***argvp, std::string *cmdp) {
  assert(*argcp > 0);
  const char *cmd = **argvp;
  while (*cmd == '-')
    ++cmd;
  ++*argvp;
  --*argcp;

  *cmdp = cmd;
}

static bool parseopt(
  int *argcp, char ***argvp,
  std::string *optp, std::string *valp
) {
  assert(*argcp >= 0);
  if (*argcp == 0)
    return false;

  const char *opt = **argvp;
  if (char *q = strchr(**argvp, '=')) {
    *q = 0;
    **argvp = q + 1;
  } else {
    if (*opt == '-' && *argcp > 1) {
      --*argcp;
      ++*argvp;
    } else {
      opt = "";
    }
  }
  while (*opt == '-')
    ++opt;

  assert(*argcp > 0);
  const char *val = **argvp;
  --*argcp;
  ++*argvp;

  *optp = opt;
  *valp = val;

  return true;
}

Cmdline::Cmdline(int *argcp, char ***argvp, const char *dopt) {
  parsecmd(argcp, argvp, &cmd);

  std::string opt, val;
  while (parseopt(argcp, argvp, &opt, &val)) {
    if (opt == "")
      opt = dopt;
    put(opt, val);
  }
}

void Cmdline::put(const std::string &opt, const std::string &val) {
  if (m.count(opt)) 
    warning(std::string("ignoring repeated option -") + opt + "=" + val);
  else {
    m[opt] = val;
    unused.insert(opt);
  }
}

const Cmdline::Value &Cmdline::get(const std::string &opt) {
  if (m.count(opt)) {
    unused.erase(opt);
    return m[opt];
  }
  error("no value for required option " + opt);
  assert(0);
}

const Cmdline::Value &Cmdline::get(const std::string &opt, const std::string &dval) {
  if (!m.count(opt))
    m[opt] = dval;
  else
    unused.erase(opt);
  return m[opt];
}

const Cmdline::Value &Cmdline::operator [](const std::string &opt) {
  return get(opt);
}

bool Cmdline::present(const std::string &opt) {
  return m.count(opt);
}

}

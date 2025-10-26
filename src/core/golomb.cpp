#include "core/golomb.hpp"
#include <algorithm>
#include <unordered_set>

namespace golomb {

bool is_valid_rule(const std::vector<int>& marks) {
  if (marks.empty()) {
    return true;
  }

  std::unordered_set<int> distances;
  for (size_t i = 0; i < marks.size(); ++i) {
    for (size_t j = i + 1; j < marks.size(); ++j) {
      int dist = marks[j] - marks[i];
      if (distances.count(dist)) {
        return false;
      }
      distances.insert(dist);
    }
  }
  return true;
}

int length(const std::vector<int>& marks) {
  if (marks.empty()) {
    return 0;
  }
  return marks.back();
}

std::vector<int> greedy_seed(int n, int ub) {
  if (n <= 0) {
    return {};
  }

  RuleState st(ub);
  st.marks.push_back(0);
  st.used.set(0);

  // Simple greedy: try positions sequentially
  int pos = 1;
  while (static_cast<int>(st.marks.size()) < n && pos < ub) {
    if (st.used.can_add_mark(st.marks, pos)) {
      st.used.add_mark(st.marks, pos);
    }
    ++pos;
  }

  return st.marks;
}

bool try_add(RuleState& st, int p) {
  if (p < 0 || p >= st.used.size()) {
    return false;
  }

  if (!st.used.can_add_mark(st.marks, p)) {
    return false;
  }

  st.used.add_mark(st.marks, p);
  return true;
}

}  // namespace golomb

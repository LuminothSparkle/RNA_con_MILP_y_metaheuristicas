#include <gurobi_c++.h>
#include <iostream>
#include <format>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <optional>
#include <utility>
#include <ranges>
#include <functional>
#include <variant>
#include <numeric>
#include <random>

namespace fsys = std::filesystem;
namespace ranges = std::ranges;
namespace views = std::views;
using fsys::path;
using std::fstream;
using std::format;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::optional;
using std::pair;
using std::getline;
using std::stoi;
using std::stod;
using std::accumulate;
using std::reference_wrapper;
using std::istream;
using std::ostream;
using std::cin;
using std::clog;
using std::cout;
using std::move;
using std::make_tuple;
using std::make_pair;
using std::tuple;
using std::variant;
using std::holds_alternative;
using std::log;
using std::log1p;
using std::exp2;
using std::random_device;
using std::bernoulli_distribution;
using std::mt19937;

template<typename T>
using vec = std::vector<T>;
template<typename T>
using mat = vec<vec<T>>;
template<typename T>
using ten3 = vec<mat<T>>;
template<typename T>
using ten4 = vec<ten3<T>>;
template<typename T>
using ten5 = vec<ten4<T>>;

bool is_single_expr(const GRBLinExpr& expr) {
   return expr.size() == 1 && expr.getConstant() == 0.0 && expr.getCoeff(0) == 1.0;
}

GRBVar gen_var(
   GRBModel& model, const GRBLinExpr& expr,
   const string& var_name, const string& constr_name,
   double min = -GRB_INFINITY, double max = GRB_INFINITY,
   char var_type = GRB_CONTINUOUS, int lazy = 0
) {
   if(is_single_expr(expr)) {
      return expr.getVar(0);
   }
   GRBVar var = model.addVar(min, max, 0, var_type, var_name);
   GRBConstr constr = model.addConstr(expr == var, constr_name);
   constr.set(GRB_IntAttr_Lazy, lazy);
   return var;
}

vec<GRBVar> gen_vars(
   GRBModel& model, const vec<GRBLinExpr>& exprs,
   const string& var_name, const string& constr_name,
   double min = -GRB_INFINITY, double max = GRB_INFINITY,
   char var_type = GRB_CONTINUOUS, int lazy = 0
) {
   vec<GRBVar> vars(exprs.size());
   for(auto&& [i, var, expr] : views::zip(views::iota(0), vars, exprs)) {
      var = gen_var(
         model,expr,
         format("{}_{}", var_name,    i),
         format("{}_{}", constr_name, i),
         min, max, var_type,
         lazy
      );
   }
   return vars;
}

GRBVar gen_abs_var(
   GRBModel& model, const GRBLinExpr& x,
   const string& var_name, const string& constr_name,
   bool use_max, bool objective, int lazy
);

GRBVar gen_abs_var(
   GRBModel& model, const GRBVar& x,
   const string& var_name, const string& constr_name,
   bool use_max = false, bool objective = false,
   int lazy = 0
) {
   if(objective || use_max) {
      return gen_abs_var(
         model, GRBLinExpr(gen_var(
            model, x,
            format("{}_vinput", var_name),
            format("{}_cinput", constr_name),
            -GRB_INFINITY, GRB_INFINITY,
            GRB_CONTINUOUS, lazy
         )),
         var_name, constr_name,
         use_max, objective,
         lazy
      );
   }
   GRBVar var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrAbs(var_abs, x, constr_name);
   return var_abs;
}

GRBVar gen_abs_var(
   GRBModel& model, const GRBLinExpr& x,
   const string& var_name, const string& constr_name,
   bool use_max = false, bool objective = false,
   int lazy = 0
) {
   if(objective || use_max) {
      GRBVar var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
      GRBConstr constr_plus = model.addConstr(x <= var_abs, format("{}_plus", constr_name));
      GRBConstr constr_minus = model.addConstr(-x <= var_abs, format("{}_minus", constr_name));
      constr_plus.set(GRB_IntAttr_Lazy, lazy);
      constr_minus.set(GRB_IntAttr_Lazy, lazy);
      if(!objective) {
         GRBVar a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or", var_name));
         model.addGenConstrIndicator(a_or_b, 1, var_abs - x, GRB_GREATER_EQUAL, 0, format("{}_on",  constr_name));
         model.addGenConstrIndicator(a_or_b, 0, var_abs + x, GRB_GREATER_EQUAL, 0, format("{}_off", constr_name));
      }
      return var_abs;
   }
   return gen_abs_var(
      model, gen_var(
         model, x,
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name,
      use_max, objective, lazy
   );
}

template<typename T>
GRBLinExpr gen_abs_expr(
   GRBModel& model, const T& x,
   const string& var_name, const string& constr_name,
   bool use_max = false, bool objective = false, int lazy = 0
) {
   return gen_abs_var(model, x, var_name, constr_name, use_max, objective, lazy);
}

GRBVar gen_max_var(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   GRBVar var_max = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   GRBVar a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or", var_name));
   GRBConstr max_a = model.addConstr(var_max >= a, format("{}_a", constr_name));
   GRBConstr max_b = model.addConstr(var_max >= b, format("{}_b", constr_name));
   max_a.set(GRB_IntAttr_Lazy, lazy);
   max_b.set(GRB_IntAttr_Lazy, lazy);
   model.addGenConstrIndicator(a_or_b, 1, var_max - a, GRB_LESS_EQUAL, 0, format("{}_on",  constr_name));
   model.addGenConstrIndicator(a_or_b, 0, var_max - b, GRB_LESS_EQUAL, 0, format("{}_off", constr_name));
   return var_max;
}

GRBLinExpr gen_max_expr(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_max_var(
      model,
      a, b,
      var_name,
      constr_name,
      lazy
   );
}

GRBVar gen_min_var(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   GRBVar var_min = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   GRBVar a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or", var_name));
   GRBConstr min_a = model.addConstr(var_min <= a, format("{}_a", constr_name));
   GRBConstr min_b = model.addConstr(var_min <= b, format("{}_b", constr_name));
   min_a.set(GRB_IntAttr_Lazy, lazy);
   min_b.set(GRB_IntAttr_Lazy, lazy);
   model.addGenConstrIndicator(a_or_b, 1, var_min - a, GRB_GREATER_EQUAL, 0, format("{}_on",  constr_name));
   model.addGenConstrIndicator(a_or_b, 0, var_min - b, GRB_GREATER_EQUAL, 0, format("{}_off", constr_name));
   return var_min;
}

GRBLinExpr gen_min_expr(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_min_var(
      model,
      a, b,
      var_name,
      constr_name,
      lazy
   );
}

GRBVar gen_max_var(
   GRBModel& model, const vec<GRBVar>& X,
   const string& var_name, const string& constr_name,
   double min = 0, int lazy = 0
) {
   GRBVar var = model.addVar(min, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMax(var, X.data(), X.size(), min, constr_name);
   return var;
}

GRBVar gen_max_var(
   GRBModel& model, const GRBLinExpr& X,
   const string& var_name, const string& constr_name,
   double min = 0, int lazy = 0
) {
   vec<GRBVar> X_vec = {
      gen_var(
         model, X,
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      )
   };
   return gen_max_var(model, X_vec, var_name, constr_name, min, lazy);
}

GRBLinExpr gen_max_expr(
   GRBModel& model, const vec<GRBVar>& X, const string& var_name,
   const string& constr_name, double min = 0, int lazy = 0
) {
   return gen_max_var(model, X, var_name, constr_name, min, lazy);
}

GRBLinExpr gen_max_expr(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double min = 0, int lazy = 0
) {
   return gen_max_var(
      model,
      gen_vars(
         model, X,
         format("{}_vinputs", var_name),
         format("{}_cinputs", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name, min, lazy
   );
}

GRBVar gen_max_var(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double min = 0, int lazy = 0
) {
   return gen_max_var(
      model,
      gen_vars(
         model,X,
         format("{}_vinputs", var_name),
         format("{}_cinputs", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name, min, lazy
   );
}

GRBVar gen_min_var(
   GRBModel& model, const vec<GRBVar>& X,
   const string& var_name, const string& constr_name,
   double max = 0, int lazy = 0
) {
   GRBVar var = model.addVar(-GRB_INFINITY, max, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMin(var, X.data(), X.size(), max, constr_name);
   return var;
}

GRBVar gen_min_var(
   GRBModel& model, const GRBLinExpr& X,
   const string& var_name, const string& constr_name,
   double max = 0, int lazy = 0
) {
   vec<GRBVar> X_vec = {
      gen_var(
         model, X,
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      )
   };
   return gen_min_var(model, X_vec, var_name, constr_name, max, lazy);
}

GRBLinExpr gen_min_expr(
   GRBModel& model, const vec<GRBVar>& X, const string& var_name,
   const string& constr_name, double max = 0, int lazy = 0
) {
   return gen_min_var(model, X, var_name, constr_name, max, lazy);
}

GRBLinExpr gen_min_expr(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double max = 0, int lazy = 0
) {
   return gen_min_expr(
      model,
      gen_vars(
         model,X,
         format("{}_vinputs", var_name),
         format("{}_cinputs", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name, max, lazy
   );
}

GRBVar gen_min_var(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double max = 0, int lazy = 0
) {
   return gen_min_var(
      model,
      gen_vars(
         model,X,
         format("{}_vinputs", var_name),
         format("{}_cinputs", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name, max, lazy
   );
}

template<typename T>
GRBLinExpr gen_sum_expr(const vec<T>& X) {
   return accumulate(X.begin(), X.end(), GRBLinExpr());
}

template<typename T>
GRBVar gen_sum_var(
   GRBModel& model, const vec<T>& X,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_var(
      model,
      gen_sum_expr(X),
      var_name,
      constr_name,
      -GRB_INFINITY,
      GRB_INFINITY,
      GRB_CONTINUOUS,
      lazy
   );
}

GRBLinExpr gen_abs_error_expr(
   GRBModel& model, const GRBLinExpr& y, const GRBLinExpr& ty,
   const string& var_name,  const string& constr_name,
   bool use_max = false, bool objective = false, int lazy = 0
) {
   return gen_abs_expr(
      model,
      y - ty,
      var_name,
      constr_name,
      use_max,
      objective,
      lazy
   );
}

GRBVar gen_abs_error_var(
   GRBModel& model, const GRBLinExpr& y, const GRBLinExpr& ty,
   const string& var_name,  const string& constr_name,
   bool use_max = false, bool objective = false, int lazy = 0
) {
   return gen_var(
      model,
      gen_abs_error_expr(
         model, y, ty, 
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         use_max, objective, lazy
      ),
      var_name, constr_name,
      0, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBVar gen_act_var(
   GRBModel& model, const T& x, const string& dropout_name,
   bool dropout = false, int lazy = 0
) {
   if(dropout) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, dropout_name);
   }
   return gen_var(
      model, x,
      format("{}_voutput", dropout_name),
      format("{}_coutput", dropout_name),
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBLinExpr gen_act_expr(
   GRBModel& model, const T& x, const string& dropout_name,
   bool dropout = false, int lazy = 0
) {
   if(dropout) {
      return GRBLinExpr();
   }
   return gen_var(
      model, x,
      format("{}_voutput", dropout_name),
      format("{}_coutput", dropout_name),
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

GRBVar gen_bin_w_var(
   GRBModel& model, const GRBVar& b, const GRBLinExpr& a,
   const string& var_name, const string& constr_name,
   double coef, bool mask, int lazy = 0
) {
   if (!mask) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, format("{}_masked", var_name));
   }
   GRBVar bw = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrIndicator(b, 1, coef * a - bw, GRB_EQUAL, 0, format("{}_on", constr_name));
   model.addGenConstrIndicator(b, 0, bw, GRB_EQUAL, 0, format("{}_off", constr_name));
   return bw;
}

GRBVar gen_bin_w_var(
   GRBModel& model, const GRBLinExpr& b, const GRBLinExpr& a,
   const string& var_name, const string& constr_name,
   double coef, bool mask, int lazy = 0
) {
   if (!mask) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, format("{}_masked", var_name));
   }
   GRBVar b_var = gen_var(
      model, b,
      format("{}_vinput", var_name),
      format("{}_cinput", constr_name),
      0, mask,
      GRB_BINARY, lazy
   );
   return gen_bin_w_var(model, b_var, a, var_name, constr_name, coef, mask, lazy);
}

GRBLinExpr gen_bin_w_expr(
   GRBModel& model, const GRBLinExpr& b, const GRBLinExpr& a,
   const string& var_name, const string& constr_name,
   double coef, bool mask, int lazy = 0
) {
   if (!mask) {
      return GRBLinExpr();
   }
   GRBVar b_var = gen_var(
      model, b,
      format("{}_vinput", var_name),
      format("{}_cinput", constr_name),
      0,mask,
      GRB_BINARY, lazy
   );
   return gen_bin_w_var(model, b_var, a, var_name, constr_name, coef, mask, lazy);
}

GRBVar gen_bin_var(GRBModel& model, const string& var_name, bool mask = true) {
   return model.addVar(0, mask, 0, GRB_BINARY, var_name);
}

GRBLinExpr gen_bin_expr(GRBModel& model, const string& var_name, bool mask = true) {
   if (!mask) {
      return GRBLinExpr();
   }
   return gen_bin_var(model, var_name, mask);
}

GRBVar gen_hardtanh_var(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name,
   const pair<double,double>& limits = {-1,1}, int lazy = 0
) {
   GRBVar ht_min = gen_min_var(
      model, z,
      limits.second,
      format("{}_vmin", var_name),
      format("{}_cmin", constr_name),
      lazy
   );
   return gen_max_var(model, ht_min, limits.first, var_name, constr_name, lazy);
}

GRBVar gen_hardtanh_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name,
   const pair<double,double>& limits = {-1,1}, int lazy = 0
) {
   return gen_hardtanh_var(
      model,
      gen_var(
         model, z,
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name, limits, lazy
   );
}

template<typename T>
GRBLinExpr gen_hardtanh_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name,
   const pair<double,double>& limits = {-1,1}, int lazy = 0
) {
   return gen_hardtanh_var(model, z, var_name, constr_name, limits, lazy);
}

GRBVar gen_hardsigmoid_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   GRBVar hs_min = gen_min_var(
      model,
      z / 6 + 0.5, 1,
      format("{}_vmin",var_name),
      format("{}_cmin",constr_name),
      lazy
   );
   return gen_max_var(model, hs_min, 0, var_name, constr_name, lazy);
}

template<typename T>
GRBLinExpr gen_hardsigmoid_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_hardsigmoid_var(model, z, var_name, constr_name, lazy);
}

GRBVar gen_ReLU6_var(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   GRBVar relu6_max = gen_max_var(
      model,
      z, 0,
      format("{}_vmax", var_name),
      format("{}_cmax", constr_name),
      lazy
   );
   return gen_min_var(model, relu6_max, 6, var_name, constr_name, lazy);
}

GRBVar gen_ReLU6_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_ReLU6_var(
      model,
      gen_var(
         model, z,
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name,
      lazy
   );
}

template<typename T>
GRBLinExpr gen_ReLU6_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_ReLU6_var(model, z, var_name, constr_name, lazy);
}

GRBVar gen_ReLU_var(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_max_var(model, z, 0, var_name, constr_name, lazy);
}

GRBVar gen_ReLU_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_ReLU_var(
      model,
      gen_var(
         model, z,
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name, lazy
   );
}

template<typename T>
GRBLinExpr gen_ReLU_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_ReLU_var(model, z, var_name, constr_name, lazy);
}

GRBLinExpr gen_LeakyReLU_expr(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name,
   double neg_coef = 0.25, int lazy = 0
) {
   GRBVar lrelu_max = gen_max_var(
      model,
      z, 0,
      format("{}_vmax", var_name),
      format("{}_cmax", constr_name),
      lazy
   );
   GRBVar lrelu_min = gen_min_var(
      model,
      z, 0,
      format("{}_vmin", var_name),
      format("{}_cmin", constr_name),
      lazy
   );
   return lrelu_max + neg_coef * lrelu_min;
}

GRBLinExpr gen_LeakyReLU_expr(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name,
   double neg_coef = 0.25, int lazy = 0
) {
   return gen_LeakyReLU_expr(
      model,
      gen_var(
         model, z,
         format("{}_vinput", var_name),
         format("{}_cinput", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      ),
      var_name, constr_name, neg_coef, lazy
   );
}

template<typename T>
GRBVar gen_LeakyReLU_var(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name,
   double neg_coef = 0.25, int lazy = 0
) {
   return gen_var(
      model,
      gen_LeakyReLU_expr(model, z, var_name, constr_name, neg_coef, lazy),
      format("{}_voutput", var_name),
      format("{}_coutput", constr_name),
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBLinExpr gen_act_w_expr(const vec<T>& bw) {
   return accumulate(bw.begin(), bw.end() - 1, GRBLinExpr(-bw.front()));
}

template<typename T>
GRBVar gen_act_w_var(
   GRBModel& model, const vec<T>& bw,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_var(
      model,
      gen_act_w_expr(bw),
      var_name, constr_name, lazy
   );
}

vec<GRBLinExpr> gen_input_exprs(
   GRBModel& model, const string& var_name,
   const vec<double> fx, const vec<bool>& dropout,
   int lazy = 0
) {
   vec<GRBLinExpr> a;
   for(const auto& [i, fx, dropout] : views::zip(views::iota(0), fx, dropout)) {
      a.emplace_back(gen_act_expr(
         model, 
         model.addVar(fx, fx, 0, GRB_CONTINUOUS, format("{}_{}", var_name, i)),
         format("drop{}_{}", var_name, i),
         dropout, lazy
      ));
   }
   return a;
}

vec<GRBVar> gen_input_vars(
   GRBModel& model, const string& var_name,
   const vec<double> fx, const vec<bool>& dropout, int lazy = 0
) {
   return gen_vars(
      model,
      gen_input_exprs(model, var_name, fx, dropout, lazy),
      format("{}_voutput", var_name),
      format("{}_coutput", var_name),
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T1, typename T2>
vec<GRBLinExpr> gen_layer_exprs(
   GRBModel& model, const vec<T1>& act, const ten3<T2>& b,
   const GRBLinExpr& bias, const string& var_name, const string& constr_name, 
   int in_C, int out_C, const mat<int>& exponent, 
   const mat<int>& mask, const vec<bool>& dropout, const mat<bool>& c_drop,
   int lazy = 0
) {
   vec<GRBLinExpr> a;
   for(const auto& [i, act, dropout] : views::zip(views::iota(0), act, dropout)) {
      a.emplace_back(gen_act_expr(
         model, act,
         format("dropz{}_{}", var_name, i),
         dropout, lazy
      ));
   }
   a.emplace_back(bias);
   mat<GRBLinExpr> aw( out_C, vec<GRBLinExpr>(in_C + 1) );
   for(const auto& [i, b, a, exponent, mask, c_drop] : views::zip( views::iota(0), b, a, exponent, mask, c_drop )) {
      for(const auto& [j, b, exponent, mask, c_drop] : views::zip( views::iota(0), b,    exponent, mask, c_drop )) {
         vec<GRBLinExpr> bw;
         if(!c_drop) {
            for(const auto& [l,b] : b | views::enumerate) {
               bw.emplace_back(gen_bin_w_expr(
                  model,
                  gen_var(
                     model, b,
                     format("{}_vinput", var_name),
                     format("{}_cinput", constr_name),
                     0, mask,
                     GRB_BINARY, lazy
                  ),
                  a,
                  format("bw{}_{}_{}_{}",   var_name,    i, j, l),
                  format("bw{}_w{},{}_D{}", constr_name, i, j, l),
                  exp2(exponent - l), mask, lazy
               ));
            }
            aw[j][i] = gen_act_w_expr(bw);
         }
      }
   }
   return aw | views::transform(gen_sum_expr<GRBLinExpr>) | ranges::to<vec<GRBLinExpr>>();
}

template<typename T>
vec<GRBVar> gen_layer_vars(
   GRBModel& model, const vec<T>& act, const ten3<GRBVar>& b,
   const GRBVar& bias, const string& var_name, const string& constr_name, 
   int in_C, int out_C, const mat<int>& precis, 
   const mat<int>& mask, const vec<double>& dropout, const mat<double>& c_drop,
   int lazy = 0
) {
   return gen_vars(
      model,
      gen_layer_exprs(
         model, act, b, bias,
         var_name, constr_name,
         in_C, out_C,
         precis, mask, dropout,
         c_drop, lazy
      ),
      format("{}_voutputs", var_name),
      format("{}_coutputs", constr_name),
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBLinExpr gen_w_expr(
   GRBModel& model, const vec<T>& b, int exponent = 4, int lazy = 0
) {
   GRBLinExpr expr = -exp2(exponent) * b.front();
   for(const auto& [l,b] : b | views::take(b.size() - 1) | views::enumerate) {
      expr += exp2(exponent - l - 1) * b;
   }
   return expr;
}

template<typename T>
GRBVar gen_w_var(
   GRBModel& model, const vec<T>& b, const string& var_name,
   const string& constr_name, int precis = 4, int lazy = 0
) {
   return gen_var(
      model,
      gen_w_expr(model, b, precis, lazy),
      var_name, constr_name,
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBLinExpr gen_l1w_expr(
   GRBModel& model, const mat<T> w,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   GRBLinExpr expr;
   for(double w_i_size = w.size(); const auto& [i, w] : w | views::enumerate) {
      for(double w_j_size = w.size(); const auto& [j, w] : w | views::enumerate) {
         expr += 1.0 / (w_i_size * w_j_size) * gen_abs_expr(
            model, w,
            format("{}_{}_{}",  var_name,    i, j),
            format("{}_w{},{}", constr_name, i, j),
            true, true, lazy
         );
      }
   }
   return expr;
}

template<typename T>
GRBVar gen_l1w_var(
   GRBModel& model, const mat<T> w,
   const string& var_name, const string& constr_name,
   int lazy = 0
) {
   return gen_var(
      model,
      gen_l1w_expr(model, w, var_name, constr_name, lazy),
      format("{}_voutput", var_name),
      format("{}_coutput", constr_name),
      0, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBLinExpr gen_l1a_expr(
   GRBModel& model, const vec<T>& a,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   GRBLinExpr expr;
   for(double coef = 1.0 / a.size(); const auto& [j,a] : a | views::enumerate) {
      expr += coef * gen_abs_expr(
         model, a,
         format("{}_{}",  var_name,    j),
         format("{}_N{}", constr_name, j),
         true, true, lazy
      );
   }
   return expr;
}

template<typename T>
GRBVar gen_l1a_var(
   GRBModel& model, const vec<T>& a,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   return gen_var(
      model, gen_l1a_expr(model, a, var_name, constr_name, lazy),
      format("{}_voutput", var_name),
      format("{}_coutput", constr_name),
      0, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
T inverse_sigmoid(T x, T zero_tolerance = 0.0001) {
   return log(x + zero_tolerance) - log1p(-x + zero_tolerance);
}

template<typename T>
GRBLinExpr gen_class_error_expr(
   GRBModel& model, const vec<T>& y, const string& var_name,
   const string& constr_name, const vec<double>& ty,
   double zero_tolerance = 0.0001, bool constraint = false,
   double constraint_tolerance = 0.1, int lazy = 0
) {
   GRBLinExpr expr;
   if(ty.size() == 1) {
      // Clasificación binaria
      if(constraint) {
         GRBConstr constr_plus = model.addConstr(
            inverse_sigmoid(
               -constraint_tolerance + ty.back(),
               zero_tolerance
            ) <= y.back(),
            format("{}_RelConstrAbsRight", constr_name)
         );
         GRBConstr constr_minus = model.addConstr(
            inverse_sigmoid(
               constraint_tolerance - ty.back(),
               zero_tolerance
            ) >= y.back(),
            format("{}_RelConstrAbsLeft", constr_name)
         );
         constr_plus.set(GRB_IntAttr_Lazy, lazy);
         constr_minus.set(GRB_IntAttr_Lazy, lazy);
      }
      else {
         expr += gen_abs_expr(
            model, y.back() - inverse_sigmoid(ty.back(), zero_tolerance),
            format("{}_abs", var_name),
            format("{}_Err",  constr_name),
            true, true, lazy
         );
      }
   }
   else {
      // Clasificación multiclase
      GRBVar c = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_C",var_name));
      for(const auto& [ty, y, cls] : views::zip( ty, y, views::iota(0) )) {
         if(constraint) {
            GRBConstr constr_plus = model.addConstr(
               log(-constraint_tolerance + ty + zero_tolerance) <= y - c,
               format("{}_RelConstrAbsRight{}", constr_name, cls)
            );
            GRBConstr constr_minus = model.addConstr(
               log(constraint_tolerance + ty + zero_tolerance) >= y - c,
               format("{}_RelConstrAbsLeft{}", constr_name, cls)
            );
            constr_plus.set(GRB_IntAttr_Lazy, lazy);
            constr_minus.set(GRB_IntAttr_Lazy, lazy);
         }
         else {
            GRBLinExpr var_expr = gen_abs_expr(
               model,
               log(zero_tolerance + ty) - y + c,
               format("{}_{}_abs", var_name,    cls),
               format("{}_Err{}",  constr_name, cls),
               true, true, lazy
            );
            expr += var_expr;
         }
      }
   }
   return expr;
}

template<typename T>
GRBVar gen_class_error_var(
   GRBModel& model, const vec<T>& y, const string& var_name,
   const string& constr_name, const vec<double>& ty,
   double zero_tolerance = 0.0001, bool constraint = false,
   double constraint_tolerance = 0.1, int lazy = 0
) {
   return gen_var(
      model,
      gen_class_error_expr(
         model, y,
         var_name, constr_name,
         ty, zero_tolerance,
         constraint,
         constraint_tolerance,
         lazy
      ),
      format("{}_voutput", var_name),
      format("{}_coutput", constr_name),
      0, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBLinExpr gen_regression_error_expr(
   GRBModel& model, const vec<T>& y, const string& var_name,
   const string& constr_name, const vec<double>& ty,
   bool constraint = false, double constraint_tolerance = 0.1,
   int lazy = 0
) {
   GRBLinExpr expr;
   for(const auto& [i, y, ty] : views::zip( views::iota(0), y, ty )) {
      if(constraint) {
         GRBConstr constr_plus = model.addConstr(
            (1.0 + constraint_tolerance) * y >= ty,
            format("{}_NConstr{}", constr_name, i)
         );
         GRBConstr constr_minus = model.addConstr(
            (1.0 - constraint_tolerance) * y <= ty,
            format("{}_NConstr{}", constr_name, i)
         );
         constr_plus.set(GRB_IntAttr_Lazy, lazy);
         constr_minus.set(GRB_IntAttr_Lazy, lazy);
      }
      else {
         GRBLinExpr var_expr = gen_abs_error_expr(
            model, y, ty,
            format("{}_{}",  var_name,    i),
            format("{}_N{}", constr_name, i),
            true, true, lazy
         );
         expr += var_expr;
      }
   }
   return expr;
}

template<typename T>
GRBVar gen_regression_error_var(
   GRBModel& model, const vec<T>& y, const string& var_name,
   const string& constr_name, const vec<double>& ty,
   bool constraint = false, double constraint_tolerance = 0.1,
   int lazy = 0
) {
   return gen_var(
      model,
      gen_regression_error_expr(
         model, y,
         var_name, constr_name,
         ty, constraint,
         constraint_tolerance, lazy
      ),
      format("{}_voutput", var_name),
      format("{}_coutput", constr_name),
      0, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
vec<GRBLinExpr> gen_activation_exprs(
   GRBModel& model, const string& type, const vec<T>& z,
   const string& var_suffix, const string constr_suffix,
   const vec<double>& LeakyReLU_coef, const pair<double,double>& hardtanh_limits,
   int lazy = 0
) {
   vec<GRBLinExpr> a;
   if(type == "ReLU") {
      for(const auto& [j,z] : z | views::enumerate) {
         a.emplace_back(gen_ReLU_expr(
            model, z,
            format("relu{}_{}",  var_suffix,    j),
            format("relu{}_N{}", constr_suffix, j),
            lazy
         ));
      }
   }
   else if(type == "ReLU6") {
      for(const auto& [j,z] : z | views::enumerate) {
         a.emplace_back(gen_ReLU6_expr(
            model, z,
            format("relu6{}_{}",  var_suffix,    j),
            format("relu6{}_N{}", constr_suffix, j),
            lazy
         ));
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      for(const auto& [j,z] : z | views::enumerate) {
         a.emplace_back(gen_LeakyReLU_expr(
            model, z,
            format("Lrelu{}_{}",  var_suffix,    j),
            format("Lrelu{}_N{}", constr_suffix, j),
            LeakyReLU_coef[j], lazy
         ));
      }
   }
   else if(type  == "Hardtanh") {
      for(const auto& [j,z] : z | views::enumerate) {
         a.emplace_back(gen_hardtanh_expr(
            model, z,
            format("ht{}_{}",  var_suffix,    j),
            format("ht{}_N{}", constr_suffix, j),
            hardtanh_limits, lazy
         ));
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& [j,z] : z | views::enumerate) {
         a.emplace_back(gen_hardsigmoid_expr(
            model, z,
            format("hs{}_{}",  var_suffix,   j),
            format("hs{}_N{}", constr_suffix, j),
            lazy
         ));
      }
   }
   else {
      for(const auto& z : z) {
         a.emplace_back(z);
      }
   } 
   return a;
}

template<typename T>
vec<GRBVar> gen_activation_vars(
   GRBModel& model, const string& type, const vec<T>& z,
   const string& var_name, const string constr_name,
   const vec<double>& LeakyReLU_coef, const pair<double,double>& hardtanh_limits,
   int lazy = 0
) {
   return gen_vars(
      model,
      gen_activation_exprs(
            model, type, z,
            var_name, constr_name,
            LeakyReLU_coef, hardtanh_limits,
            lazy
      ),
      format("{}_voutputs", var_name),
      format("{}_coutputs", constr_name),
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

template<typename T>
GRBLinExpr gen_wp_expr(
   GRBModel& model, const mat<T>& w, const mat<double>& tw,
   const string& var_name, const string& constr_name, int lazy = 0
) {
   GRBLinExpr expr;
   for(double w_i_size = w.size(); const auto& [i, w, tw] : views::zip(views::iota(0), w, tw)) {
      for(double w_j_size = w.size(); const auto& [j, w, tw] : views::zip(views::iota(0), w, tw)) {
         expr += 1.0 / (w_i_size * w_j_size) * gen_abs_error_expr(
            model, w, tw,
            format("{}_{}_{}",  var_name,    i, j),
            format("{}_w{},{}", constr_name, i, j),
            true, true, lazy
         );
      }
   }
   return expr;
}

GRBModel get_model(
   const GRBEnv& environment, int T, int L, const vec<int>& C, const vec<string>& AF, 
   const vec<mat<int>>& mask, const vec<mat<int>>& precis, const vec<mat<int>>& D,
   const vec<mat<double>>& tw, const vec<double>& bias_w, const mat<double>& leakyReLU_coef,
   const vec<pair<double,double>>& hardtanh_limits,
   const vec<vec<bool>>& dropout,
   const vec<optional<double>>& l1a_norm,
   const vec<optional<double>>& l1w_norm,
   const vec<optional<double>>& w_pen,
   const vec<mat<bool>>& c_drop,
   const mat<double>& fx, const mat<double>& reg_ty, const mat<double>& class_ty,
   double zero_tolerance = 0.0001, double constr_frac = 0.0,
   double constraint_tolerance = 0.1, int lazy = 1
) {
   GRBModel model(environment);
   ten4<GRBLinExpr> b(L);
   vec<GRBLinExpr> bias(L);
   GRBLinExpr L1_expr;
   GRBLinExpr wp_expr;
   for(int k = 0; k < L; ++k) {
      cout << "Processing layer: " << k << "\n";
      b[k] = ten3<GRBLinExpr>(C[k] + 1,mat<GRBLinExpr>(C[k + 1]));
      mat<GRBLinExpr> w(C[k] + 1,vec<GRBLinExpr>(C[k + 1]));
      for(int i = 0; i <= C[k]; ++i) {
         for(int j = 0; j < C[k + 1]; ++j) {
            b[k][i][j] = vec<GRBLinExpr>(D[k][i][j] + 1);
            for(int l = 0; l <= D[k][i][j]; ++l) {
               b[k][i][j][l] = gen_bin_expr(model, format("b_{}_{}_{}_{}", k, i, j, l), mask[k][i][j]);
            }
            w[i][j] = gen_w_var(
               model, b[k][i][j],
               format("w_{}_{}_{}", k, i, j), format("w_{}_{}_{}", k, i, j),
               precis[k][i][j], 0
            );
         }
      }
      if(w_pen[k]) {
         wp_expr += *w_pen[k] * gen_wp_expr(model, w, tw[k], format("l1w_{}", k), format("l1w_L{}", k), lazy);
      }
      if(l1w_norm[k]) {
         L1_expr += *l1w_norm[k] * gen_l1w_expr(model, w, format("l1w_{}", k), format("l1w_L{}", k), lazy);
      }
      bias[k] = model.addVar(bias_w[k], bias_w[k], 0, GRB_CONTINUOUS, format("bias_{}", k));
   }
   int constr_size = constr_frac * fx.size();
   GRBLinExpr EC_expr, ER_expr;
   for(const auto& [t, fx, tc_y, treg_y] : views::zip(views::iota(0), fx, class_ty, reg_ty)) {
      cout << "Processing case: " << t << "\n";
      vec<GRBLinExpr> a = gen_input_exprs(model, format("x_{}",t), fx, dropout.back(), lazy);
      for(const auto& [k, b, AF, D, precis, bias, dropout, leakyReLU_coef, hardtanh_limits, c_drop] : views::zip(
         views::iota(0), b, AF, D, precis, bias, dropout | views::drop(1), leakyReLU_coef, hardtanh_limits, c_drop
      )) {
         cout << "Processing layer: " << k << "\n";
         const vec<GRBLinExpr>& z = gen_layer_exprs(
            model, a, b, bias,
            format("_{}_{}", t, k), format("_L{}_C{}", k, t),
            C[k], C[k + 1], precis,
            mask[k], dropout, c_drop,
            lazy
         );
         a = gen_activation_exprs(
            model, AF, z,
            format("_{}_{}", t, k), format("_L{}_C{}", k, t),
            leakyReLU_coef, hardtanh_limits,
            lazy
         );
         if(l1a_norm[k]) {
            L1_expr += *l1a_norm[k] * gen_l1a_expr(
               model, a,
               format("l1a_{}_{}", t, k), format("l1a_L{}_C{}", k, t),
               lazy
            );
         }
      }
      const auto& ry_view = a | views::take(treg_y.size()) | ranges::to<vec<GRBLinExpr>>();
      const auto& cy_view = a | views::drop(treg_y.size()) | ranges::to<vec<GRBLinExpr>>();
      EC_expr += gen_class_error_expr(
         model, cy_view,
         format("EC_{}", t), format("ClassE_{}", t),
         tc_y, zero_tolerance, t < constr_size, constraint_tolerance,
         lazy
      );
      ER_expr += gen_regression_error_expr(
         model, ry_view,
         format("ER_{}", t), format("RegE_{}", t),
         treg_y, t < constr_size, constraint_tolerance,
         lazy
      );
   }
   model.setObjective(EC_expr + ER_expr + L1_expr + wp_expr, GRB_MINIMIZE);
   return model;
}

template<typename G>
auto generate_dropouts(
   const vec<int>& capacity,
   const vec<optional<double>>& dropout,
   const vec<optional<double>>& connection_dropout,
   G&& generator
) {
   vec<vec<bool>> dropout_bool(dropout.size());
   vec<mat<bool>> c_dropout_bool(connection_dropout.size());
   for(auto&& [db, pd, cdb, cpd, li] : views::zip(
      dropout_bool, dropout, c_dropout_bool, connection_dropout, capacity | views::slide(2)
   )) {
      const auto n = li[0], m = li[1];
      db = views::repeat(pd, n) | views::transform(
         [&generator](const optional<double>& dropout){
            return dropout.transform([&generator](double dropout) {
               return bernoulli_distribution(dropout)(generator);
            }).value_or(false);
         }
      ) | ranges::to<vec<bool>>();
      cdb = views::repeat(cpd, n + 1) | views::transform(
         [&generator, &m](const optional<double>& dropout) {
            return views::repeat(dropout, m) | views::transform(
               [&generator](const optional<double>& dropout) {
                  return dropout.transform([&generator](double dropout) {
                     return bernoulli_distribution(dropout)(generator);
                  }).value_or(false);
               }
            ) | ranges::to<vec<bool>>();
         }
      ) | ranges::to<mat<bool>>();
   }
   return make_tuple(dropout_bool, c_dropout_bool);
}

template<typename T>
istream& operator>>(istream& stream, optional<T>& opt) {
   T value; stream >> value; opt = value;
   return stream;
}

template<typename T>
ostream& operator<<(ostream& stream, const optional<T>& opt) {
   if(opt.has_value()) {
      return stream << *opt;
   }
   return stream << "";
}

template<typename T>
auto read_matrix_from_csv(istream&& input, bool ignore_header = false, bool ignore_index = false) {
   string line, word;
   stringstream line_stream, word_stream;
   if(ignore_header) {
      getline(input, line);
   }
   mat<T> matrix;
   while(getline(input, line)) {
      line_stream = stringstream(line);
      vec<T> vector;
      if(ignore_index) {
         getline(line_stream, word, ',');
      }
      while(getline(line_stream, word, ',')) {
         word_stream = stringstream(word);
         T value;
         word_stream >> value;
         vector.emplace_back(value);
      }
      matrix.emplace_back(vector);
   }
   return matrix;
}

template<typename T>
auto read_list_from_csv(istream&& input, bool ignore_index = false) {
   string line, word;
   stringstream line_stream, word_stream;
   getline(input, line);
   line_stream = stringstream(line);
   if(ignore_index) {
      getline(line_stream, word, ',');
   }
   int max_dim = 0;
   while(getline(line_stream, word, ',') && word.starts_with("d_")) {
      ++max_dim;
   }
   vec<vec<int>> dim_list;
   vec<vec<T>> data_list;
   while(getline(input, line)) {
      line_stream = stringstream(line);
      if(ignore_index) {
         getline(line_stream, word, ',');
      }
      vec<int> dim(max_dim);
      for(auto& d : dim) {
         getline(line_stream, word, ',');
         word_stream = stringstream(word);
         word_stream >> d;
      }
      vec<T> data;
      while(getline(line_stream, word, ',') && word != "") {
         word_stream = stringstream(word);
         T value;
         word_stream >> value;
         data.emplace_back(value);
      }
      dim_list.emplace_back(dim);
      data_list.emplace_back(data);
   }
   return make_tuple(dim_list, data_list);
}

auto read_arch(istream&& input, bool ignore_index = false, bool ignore_header = true) {
   string line, word;
   stringstream line_stream, word_stream;
   if(ignore_header) {
      getline(input, line);
   }
   vec<int> C;
   vec<string> AF;
   vec<double> bias;
   vec<pair<double,double>> HT;
   vec<optional<double>> Drop, L1w, L1a, w_pen, c_drop;
   while(getline(input, line)) {
      line_stream = stringstream(line);
      if(ignore_index) {
         getline(line_stream, word, ',');
      }
      int k;
      optional<double> l1a, l1w, drop, b, ht_min, ht_max, wp, cd;
      optional<string> af;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> k;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> af;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> drop;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> ht_min;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> ht_max;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> l1w;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> l1a;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> b;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> wp;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> cd;
      C.emplace_back(k);
      AF.emplace_back(af.value_or("None"));
      HT.emplace_back(make_pair(ht_min.value_or(-1), ht_max.value_or(1)));
      Drop.emplace_back(drop);
      L1w.emplace_back(l1w);
      L1a.emplace_back(l1a);
      bias.emplace_back(b.value_or(1.0));
      w_pen.emplace_back(wp);
      c_drop.emplace_back(cd);
   }
   return make_tuple(C, AF, HT, Drop, L1w, L1a, bias, w_pen, c_drop);
}

template<typename T>
auto get_layers_matrix(const mat<int>& dim, const vec<vec<T>>& data) {
   vec<mat<T>> layers_data;
   for(const auto& [dim, data] : views::zip(dim, data)) {
      int n = dim[0], m = dim[1];
      mat<T> layer_data(n, vec<T>(m));
      for(int i = 0; i < n; ++i) {
         for(int j = 0; j < m; ++j) {
            layer_data[i][j] = data[i * m + j];
         }
      }
      layers_data.emplace_back(layer_data);
   }
   return layers_data;
}

variant< unordered_map< string, vec<string> >, string > process_opts(int argc, const char* argv[]) {
   const unordered_map< string, int > opts = {
      {"load_path", 1},
      {"load_name", 1},
      {"save_path", 1},
      {"save_name", 1},
      {"time_limit", 1},
      {"solution_limit", 1},
      {"iteration_limit", 1},
      {"node_limit", 1},
      {"opt_tol", 1},
      {"best_obj_stop", 1},
      {"feas_tol", 1},
      {"int_feas_tol", 1},
      {"no_log_to_console", 0},
      {"no_save_sols", 0},
      {"no_save_log", 0},
      {"no_save_json", 0},
      {"no_save_sol", 0},
      {"no_save_mst", 0},
      {"no_save_ilp", 0},
      {"no_save_lp", 0},
      {"no_optimize", 0},
      {"use_precision", 0},
      {"use_bias", 0},
      {"use_mask", 0},
      {"use_leakyReLU", 0},
      {"use_bits", 0},
      {"zero_tolerance", 1},
      {"constr_tol", 1},
      {"constr_frac", 1},
      {"mst_file", 1},
      {"use_init", 1},
      {"use_penalty", 1},
      {"lazy", 1},
      {"seed", 1}
   };
   unordered_map< string, vec<string> > processed_opts;
   int argi = 1;
   while(argi < argc) {
      if(string arg = argv[argi]; !arg.starts_with("--")) {
         return format("Error: {} no es una opcion", arg);
      }
      else if(string arg_name = arg.substr(2); !opts.contains(arg_name)) {
         return format("Error: No existe la opcion {}", arg_name);
      }
      else if(int max_argi = argi + opts.at(arg_name); max_argi >= argc) {
         return format("Error: Faltaron {} argumentos", max_argi - argc + 1);
      }
      else {
         processed_opts[arg_name] = vec<string>();
         for(++argi; argi <= max_argi; ++argi) {
            if(arg = argv[argi]; arg.starts_with("--")) {
               return format(
                  "Error: La opcion {} se puso antes de todos los argumentos de la opcion anterior",
                  arg
               );
            }
            else {
               processed_opts[arg_name].emplace_back(arg);
            } 
         }
      }
   }
   return processed_opts;
}

template<typename Func>
void process_yes_arg(
   const unordered_map< string, vec<string> >& opts,
   const string& arg_name, Func f
) {
   if(opts.contains(arg_name)) {
      f(opts.at(arg_name));
   }
}

template<typename Func>
void process_no_arg(
   const unordered_map< string, vec<string> >& opts,
   const string& arg_name, Func f
) {
   if(!opts.contains(arg_name)) {
      f();
   }
}

string safe_suffix(const string & a, const string& b) {
   if(!a.ends_with("_") && b.compare("") != 0) {
      return a + "_" + b;
   }
   return a + b;
}

int main(int argc, const char* argv[]) try {
   auto e_opts = process_opts(argc,argv);
   if(holds_alternative<string>(e_opts)) {
      cout << get<string>(e_opts);
      return 0;
   }
   auto opts = get<unordered_map<string,vec<string>>>(e_opts);
   path save_path = opts.contains("save_path") ? opts["save_path"][0] : "";
   path load_path = opts.contains("load_path") ? opts["load_path"][0] : "";
   string save_name = opts.contains("save_name") ? opts["save_name"][0] : "model";
   string load_name = opts.contains("load_name") ? opts["load_name"][0] : "";
   path arch_path = load_path / format("{}.csv", safe_suffix(load_name, "arch"));
   path features_path = load_path / format("{}.csv", safe_suffix(load_name, "ftr"));
   path class_targets_path = load_path / format("{}.csv", safe_suffix(load_name, "cls_tgt"));
   path regression_targets_path = load_path / format("{}.csv", safe_suffix(load_name, "reg_tgt"));
   const auto& [C, AF, hardtanh, dropout, l1w_norm, l1a_norm, bias, w_pen, c_drop] = read_arch(fstream(arch_path));
   const auto& features = read_matrix_from_csv<double>(fstream(features_path), true, true);
   auto regression_targets = read_matrix_from_csv<double>(fstream(regression_targets_path), true, true);
   auto class_targets = read_matrix_from_csv<double>(fstream(class_targets_path), true, true);
   int T = features.size(), L = C.size() - 1;
   if(class_targets.empty()) {
      class_targets = mat<double>(T);
   }
   if(regression_targets.empty()) {
      regression_targets = mat<double>(T);
   }
   vec<mat<int>> bits;
   if(opts.contains("use_bits")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "bits"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() == 2) {
         bits = get_layers_matrix(dim, data);
      }
   }
   if(bits.empty()) {
      bits = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         bits[k] = mat<int>(C[k] + 1, vec<int>(C[k + 1]));
         for(auto& row : bits[k]) {
            for(auto& value : row) {
               value = 4;
            }
         }
      }
   }
   vec<mat<int>> precision;
   if(opts.contains("use_precision")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "exp"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() == 2) {
         precision = get_layers_matrix(dim, data);
      }
   }
   if(precision.empty()) {
      precision = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         precision[k] = mat<int>(C[k] + 1, vec<int>(C[k + 1]));
         for(auto& row : precision[k]) {
            for(auto& value : row) {
               value = 2;
            }
         }
      }
   }
   vec<mat<int>> mask;
   if(opts.contains("use_mask")) {
      path file_path = load_path / format("{}.csv" ,safe_suffix(load_name, "mask"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() == 2) {
         mask = get_layers_matrix(dim, data);
      }
   }
   if(mask.empty()) {
      mask = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         mask[k] = mat<int>(C[k] + 1, vec<int>(C[k + 1]));
         for(auto& row : mask[k]) {
            for(auto& value : row) {
               value = 1;
            }
         }
      }
   }
   vec<mat<double>> init_w;
   if(opts.contains("use_init")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "init"));
      const auto& [dim, data] = read_list_from_csv<double>(fstream(file_path));
      if(dim.size() == 2) {
         init_w = get_layers_matrix(dim, data);
      }
   }
   if(init_w.empty()) {
      init_w = vec<mat<double>>(L);
      for(int k = 0; k < L; ++k) {
         init_w[k] = mat<double>(C[k] + 1, vec<double>(C[k + 1]));
         for(auto& row : init_w[k]) {
            for(auto& value : row) {
               value = 0;
            }
         }
      }
   }
   vec<vec<double>> leakyReLU;
   if(opts.contains("use_leakyReLU")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "lReLU"));
      const auto& [dim, data] = read_list_from_csv<double>(fstream(file_path));
      leakyReLU = data;
   }
   else {
      leakyReLU = vec<vec<double>>(L);
      for(int k = 0; k < L; ++k) {
         leakyReLU[k] = vec<double>(C[k + 1]);
         for(auto& value : leakyReLU[k]) {
            value = 0.25;
         }
      }
   }
   string file_path = (save_path / safe_suffix(load_name, save_name)).string();
   string ResultFile = format("{}.sol.7z", file_path);
   string SolFiles = file_path;
   string LogFile = format("{}.log", file_path);
   GRBEnv ambiente;
   stringstream argstream;
   int LogToConsole = !opts.contains("no_log_to_console");
   ambiente.set(GRB_IntParam_LogToConsole, LogToConsole);
   process_no_arg(opts, "no_save_sols", [&SolFiles, &ambiente]() {
      ambiente.set(GRB_StringParam_SolFiles, SolFiles);
   });
   process_no_arg(opts, "no_save_log", [&LogFile, &ambiente]() {
      ambiente.set(GRB_StringParam_LogFile, LogFile);
   });
   process_yes_arg(opts, "best_obj_stop", [&argstream, &ambiente](const auto& args) {
      double BestObjStop;
      argstream = stringstream(args[0]);
      argstream >> BestObjStop;
      ambiente.set(GRB_DoubleParam_BestObjStop, BestObjStop);
   });
   unsigned int seed = random_device()();
   process_yes_arg(opts, "seed", [&argstream, &seed](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> seed;
   });
   clog << format("Seed used: {}", seed);
   double zero_tolerance = 0.0001;
   process_yes_arg(opts, "zero_tolerance", [&argstream, &zero_tolerance](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> zero_tolerance;
   });
   double constraint_tolerance = 0.1;
   process_yes_arg(opts, "constr_tol", [&argstream, &constraint_tolerance](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> constraint_tolerance;
   });
   double constraint_frac = 0.0;
   process_yes_arg(opts, "constr_tol", [&argstream, &constraint_frac](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> constraint_frac;
   });
   int lazy = 1;
   process_yes_arg(opts, "lazy", [&argstream, &lazy](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> lazy;
   });
   process_yes_arg(opts, "feas_tol", [&argstream, &ambiente](const auto& args) {
      double FeasibilityTol;
      argstream = stringstream(args[0]);
      argstream >> FeasibilityTol;
      ambiente.set(GRB_DoubleParam_FeasibilityTol, FeasibilityTol);
   });
   process_yes_arg(opts, "int_feas_tol", [&argstream, &ambiente](const auto& args) {
      double IntFeasTol;
      argstream = stringstream(args[0]);
      argstream >> IntFeasTol;
      ambiente.set(GRB_DoubleParam_IntFeasTol, IntFeasTol);
   });
   process_yes_arg(opts, "iteration_limit", [&argstream, &ambiente](const auto& args) {
      double IterationLimit;
      argstream = stringstream(args[0]);
      argstream >> IterationLimit;
      ambiente.set(GRB_DoubleParam_IterationLimit, IterationLimit);
   });
   process_yes_arg(opts, "opt_tol", [&argstream, &ambiente](const auto& args) {
      double OptimalityTol;
      argstream = stringstream(args[0]);
      argstream >> OptimalityTol;
      ambiente.set(GRB_DoubleParam_OptimalityTol, OptimalityTol);
   });
   process_yes_arg(opts, "solution_limit", [&argstream, &ambiente](const auto& args) {
      int SolutionLimit;
      argstream = stringstream(args[0]);
      argstream >> SolutionLimit;
      ambiente.set(GRB_IntParam_SolutionLimit, SolutionLimit);
   });
   process_yes_arg(opts, "time_limit", [&argstream, &ambiente](const auto& args) {
      double TimeLimit;
      argstream = stringstream(args[0]);
      argstream >> TimeLimit;
      ambiente.set(GRB_DoubleParam_TimeLimit, TimeLimit);
   });
   process_yes_arg(opts, "node_limit", [&argstream, &ambiente](const auto& args) {
      double NodeLimit;
      argstream = stringstream(args[0]);
      argstream >> NodeLimit;
      ambiente.set(GRB_DoubleParam_NodeLimit, NodeLimit);
   });
   int JSONSolDetail = 1;
   ambiente.set(GRB_IntParam_JSONSolDetail, JSONSolDetail);
   bool optimize = !opts.contains("no_optimize");
   bool save_lp = !opts.contains("no_save_lp");
   bool save_ilp = !opts.contains("no_save_ilp"); 
   bool save_sol = !opts.contains("no_save_sol");
   bool save_mst = !opts.contains("no_save_mst");
   bool save_json = !opts.contains("no_save_json");
   const auto& [db, cdb] = generate_dropouts(C, dropout, c_drop, mt19937(seed));
   GRBModel modelo = get_model(
      ambiente, T, L, C, AF,
      mask, precision, bits, init_w, bias,
      leakyReLU, hardtanh, db, l1a_norm, l1w_norm, w_pen, cdb,
      features, class_targets, regression_targets,
      zero_tolerance, constraint_frac, constraint_tolerance,
      lazy
   );
   if(save_lp)
      modelo.write(path(format("{}.lp.7z", file_path)).string());
   if(optimize) {
      process_yes_arg(opts, "mst_file", [&argstream, &modelo](const auto& args) {
         path mst_path(args[0]);
         modelo.read(mst_path.string());
      });
      modelo.update();
      modelo.optimize( );
      int Status = modelo.get(GRB_IntAttr_Status);
      switch(Status) {
         case GRB_OPTIMAL :
            cout << "Solución encontrada\n";
         case GRB_SUBOPTIMAL :
         case GRB_ITERATION_LIMIT :
         case GRB_NODE_LIMIT :
         case GRB_TIME_LIMIT :
         case GRB_INTERRUPTED :
            if(save_sol) {
               modelo.write(format("{}.sol.7z", file_path));
            }
            if(save_json) {
               modelo.write(format("{}.json.7z", file_path));
            } 
            if(save_mst) {
               modelo.write(format("{}.mst.7z", file_path));
            }
            break;
         case GRB_INFEASIBLE :
            cout << "Modelo infactible\n";
            if(save_ilp) {
               modelo.computeIIS( );
               modelo.write(format("{}.ilp.7z", file_path));
            }
            break;
         case GRB_UNBOUNDED :
            cout << "Modelo no acotado\n";
            break;
         default :
            cout << "Estado no manejado\n";
      }
      cout << format("Tiempo tardado: {}s\n", modelo.get(GRB_DoubleAttr_Runtime));
   }
} catch (const GRBException& ex) {
   cout << ex.getMessage( ) << "\n";
}

// g++ programa.cpp -std=c++23 -lgurobi_c++ -lgurobi110 -o programa
// ./programa --load_path [load_path] --save_path [save_path] --save_name [save_name] --load_name [load_name]
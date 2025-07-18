#include <gurobi_c++.h>
#include <iostream>
#include <iomanip>
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
#include <array>
#include <concepts>
#include <type_traits>

// Simplificacion de espacio de nombres
namespace fsys = std::filesystem;
namespace ranges = std::ranges;
namespace views = std::views;
// Manejo de rutas
using fsys::path;
using fsys::directory_iterator;
// Funciones y tipos de cadenas
using std::format;
using std::string;
using std::stoi;
using std::stod;
// Algoritmos de conveniencia
using std::accumulate;
// Funciones de utilidad
using std::reference_wrapper;
using std::move;
using std::forward;
// Contenedores o utilidades
using std::make_tuple;
using std::make_pair;
using std::unordered_map;
using std::optional;
using std::pair;
using std::tuple;
using std::variant;
using std::holds_alternative;
using std::monostate;
using std::array;
// Streams y relacionados utilizados
using std::flush;
using std::cin;
using std::clog;
using std::cout;
using std::stringstream;
using std::istringstream;
using std::fstream;
using std::istream;
using std::ostream;
using std::getline;
// Funciones de random utilizadas
using std::uniform_int_distribution;
using std::random_device;
using std::bernoulli_distribution;
using std::mt19937;
// Funciones de cmath utilizadas
using std::pow;
using std::log10;
using std::fma;
using std::log;
using std::log1p;
using std::exp2;
using std::exp;
using std::abs;
using std::frexp;
using std::ldexp;
using std::modf;
using std::max;
using std::min;
// Conceptos usados
using std::same_as;

// Simplificacion de tipos de datos a tensores
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

template<typename T>
concept GRBLinExprAcceptable = requires (T r) {
   GRBLinExpr() + r;
};

template<typename R>
concept AlgorithmicRange = (
   ranges::common_range<R> && ranges::viewable_range<R> &&
   ranges::forward_range<R> && ranges::input_range<R>
);

template<typename R, typename T>
concept RangeOf = same_as<ranges::range_value_t<R>, T> && AlgorithmicRange<R>;

template<typename R, typename T>
concept MatrixRange = RangeOf<ranges::range_value_t<R>, T> && AlgorithmicRange<R>;

template<typename R, typename T>
concept Tensor3Range = MatrixRange<ranges::range_value_t<R>, T> && AlgorithmicRange<R>;

template<typename R, typename T>
concept Tensor4Range = Tensor3Range<ranges::range_value_t<R>, T> && AlgorithmicRange<R>;

template<typename R, typename T>
concept Tensor5Range = Tensor4Range<ranges::range_value_t<R>, T> && AlgorithmicRange<R>;

template<typename R>
concept GRBLinExprRange = GRBLinExprAcceptable< ranges::range_value_t<R> > && AlgorithmicRange<R>;

void resetline_console() {
   cout << "\x0D\x1b[K" << flush;
}

void cursorup_console(int n) {
   cout << "\x1b[" << n << "F" << flush;
}

template<typename T>
T inverse_sigmoid(T x, T tol0 = 0.000000000001) {
   return log(max(x, tol0)) - log1p(max(-x, tol0 - 1));
}

string safe_suffix(const string & a, const string& b) {
   if(!a.ends_with("_") && b.compare("") != 0) {
      return a + "_" + b;
   }
   return a + b;
}

template<typename T>
istream& operator >>(istream& stream, optional<T>& opt) {
   T value; stream >> value;
   if(stream) {
      opt = value;
   }
   return stream;
}

template<typename T>
istream&& operator >>(istream&& stream, optional<T>& opt) {
   stream >> opt;
   return static_cast<decltype(stream)>(stream);
}

template<typename T>
ostream& operator <<(ostream& stream, const optional<T>& opt) {
   if(opt.has_value()) {
      return stream << *opt;
   }
   return stream;
}

template<typename T>
ostream&& operator <<(ostream&& stream, const optional<T>& opt) {
   return move(stream << opt);
}

auto decompose_w(const optional<double>& w, int size) {
   vec<optional<double>> bits_floor(size), bits_ceil(size);
   if(w) {
      int exp, sign = *w == 0 ? 0 : (1 - 2 * (*w < 0));
      double imantissa = frexp(*w, &exp) * exp2(size - 1);
      int inf = floor(imantissa * -sign) * sign;
      int sup = ceil( imantissa * -sign) * sign;
      for(auto&& [bf, bc] : views::zip(bits_floor, bits_ceil) | views::reverse) {
         bf = inf & 1; bc = sup & 1;
         inf >>= 1;    sup >>= 1;
      }
   }
   return make_tuple(bits_floor, bits_ceil);
}

double calculate_w(const RangeOf<optional<double>> auto& b, int exp) {
   double result = ldexp(-(*ranges::begin(b)).value_or(0), exp);
   for(const auto& [i, b] : b | views::enumerate | views::drop(1)) {
      result = fma(b.value_or(0), exp2(exp - i), result);
   }
   return result;
}

void set_grb_int_param(GRBEnv& env, const string& arg, GRB_IntParam code) {
   int param; stringstream(arg) >> param;
   env.set(code, param);
}

void set_grb_double_param(GRBEnv& env, const string& arg, GRB_DoubleParam code) {
   double param; stringstream(arg) >> param;
   env.set(code, param);
}

void set_grb_string_param(GRBEnv& env, const string& arg, GRB_StringParam code) {
   string param; stringstream(arg) >> param;
   env.set(code, param);
}

bool is_single_expr(const GRBLinExpr& expr) {
   return expr.size() == 1 && expr.getConstant() == 0.0 && expr.getCoeff(0) == 1.0;
}

void set_hint(GRBVar& var, double value, int prio = -10) {
   var.set(GRB_DoubleAttr_VarHintVal, value);
   var.set(GRB_IntAttr_VarHintPri,    prio );
}

void set_start(GRBModel& model, GRBVar& var, int start, double value) {
   model.set(GRB_IntParam_StartNumber, start);
   var.set(  GRB_DoubleAttr_Start,     value);
}

void gen_range_constr(
   GRBModel& model,    const GRBLinExpr& expr, double lb, double ub,
   const string& name, int lazy = 1,           bool use_grb = false
) {
   if(use_grb) {
      model.addRange(expr, lb, ub, name).set(GRB_IntAttr_Lazy, lazy);
   }
   else {
      const auto& var_u = model.addVar(0, ub - lb, 0, GRB_CONTINUOUS, format("{}_u", name));
      model.addConstr(var_u + expr == ub, name).set(GRB_IntAttr_Lazy, lazy);
   }
}

auto gen_abs_obj_vars(GRBModel& model, const string& name, int phint = -10, bool hint = true) {
   GRBVar plus  = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_plus", name));
   GRBVar minus = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_minus", name));
   if(hint) {
      set_hint(plus,  0, phint); set_hint(minus, 0, phint);
   }
   return make_tuple(plus, minus);
}

auto gen_abs_range_obj_vars(
   GRBModel& model,      const GRBLinExpr& expr, const string& name,
   double lb = 0,        double ub = 0,          int lazy = 1,
   bool use_grb = false, int phint = -10,        bool hint = true
) {
   const auto& [plus, minus] = gen_abs_obj_vars(model, name, phint, hint);
   gen_range_constr(model, minus - plus + expr, lb, ub, name, lazy, use_grb);
   return make_tuple(plus, minus);
}

GRBLinExpr gen_abs_range_obj_expr(
   GRBModel& model,      const GRBLinExpr& expr, const string& name,
   double lb = 0,        double ub = 0,          int lazy = 1,
   bool use_grb = false, int phint = -10,        bool hint = true
) {
   const auto& [plus, minus] = gen_abs_range_obj_vars(
      model, expr, name, lb, ub, lazy, use_grb, phint, hint
   );
   return plus + minus;
}

auto gen_approx_square_obj_exprs(
   GRBModel& model, const GRBLinExpr& expr, const string& name,
   int lazy = 1,    bool use_grb = false,   int phint = -10,    bool hint = true
) {
   auto&& [p1, m1] = gen_abs_obj_vars(model, format("{}_1", name), phint, hint);
   auto&& [p2, m2] = gen_abs_obj_vars(model, format("{}_2", name), phint, hint);
   auto&& [p3, m3] = gen_abs_obj_vars(model, format("{}_3", name), phint, hint);
   p1.set(GRB_DoubleAttr_UB, 2.0); m1.set(GRB_DoubleAttr_UB, 2.0);
   p2.set(GRB_DoubleAttr_UB, 4.5); m2.set(GRB_DoubleAttr_UB, 4.5);
   gen_range_constr(model, m1 - p1 + m2 - p2 + m3 - p3 + expr, -0.5, 0.5, name, lazy, use_grb);
   return make_tuple(2 * p1 + 8 * p2 + 20 * p3,  2 * m1 + 8 * m2 + 20 * m3);
}

GRBLinExpr gen_approx_square_obj_expr(
   GRBModel& model, const GRBLinExpr& expr, const string& name,
   int lazy = 1,    bool use_grb = false,   int phint = -10,    bool hint = true
) {
   const auto& [plus, minus] = gen_approx_square_obj_exprs(model, expr, name, lazy, use_grb, phint, hint);
   return plus + minus;
}

GRBLinExpr gen_sum_expr(const GRBLinExprRange auto& X) {
   return accumulate(ranges::begin(X), ranges::end(X), GRBLinExpr());
}

GRBLinExpr gen_mean_expr(const GRBLinExprRange auto& X) {
   return gen_sum_expr(X) / max<int>(1, ranges::distance(X));
}

GRBVar gen_var(
   GRBModel& model,           const GRBLinExpr& expr,    const string& name,
   double lb = -GRB_INFINITY, double ub = GRB_INFINITY, char vtype = GRB_CONTINUOUS,
   int lazy = 0
) {
   if(is_single_expr(expr)) {
      return expr.getVar(0);
   }
   GRBVar var = model.addVar(lb, ub, 0, vtype, name);
   model.addConstr(expr == var, name).set(GRB_IntAttr_Lazy, lazy);
   return var;
}

GRBVar gen_abs_var(
   GRBModel& model,      const GRBLinExpr& x, const string& name,
   bool use_grb = false, bool obj = false,    int lazy = 0,   int phint = -10,    bool hint = true, bool use_bound = false, double bound = 0.0
) {
   GRBVar var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
   if(use_grb) {
      const auto& new_x = gen_var(
         model, x, format("{}_in", name),
         -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, lazy
      );
      model.addGenConstrAbs(var_abs, new_x, name);
   }
   else {
      if(obj) {
         model.addConstr(var_abs >=  x, format("{}_plus",  name)).set(GRB_IntAttr_Lazy, lazy);
         model.addConstr(var_abs >= -x, format("{}_minus", name)).set(GRB_IntAttr_Lazy, lazy);
         if(hint) {
            set_hint(var_abs, phint, 0);
         }
      }
      else if(use_bound) {
         const auto& a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or", name));
         model.addConstr(                      x <= var_abs, format("{}_mlb", name)).set(GRB_IntAttr_Lazy, lazy);
         model.addConstr(                         -x <= var_abs, format("{}_plb", name)).set(GRB_IntAttr_Lazy, lazy);
         model.addConstr(bound *      a_or_b  + x >= var_abs, format("{}_mub", name)).set(GRB_IntAttr_Lazy, lazy);
         model.addConstr(bound * (1 - a_or_b) - x >= var_abs, format("{}_pub", name)).set(GRB_IntAttr_Lazy, lazy);
      }
      else {
         const auto& a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or", name));
         model.addGenConstrIndicator(a_or_b, 1, var_abs ==  x, format("{}_on",  name));
         model.addGenConstrIndicator(a_or_b, 0, var_abs == -x, format("{}_off", name));
      }
   }
   return var_abs;
}

GRBLinExpr gen_abs_expr(
   GRBModel& model, const GRBLinExpr& x, const string& name,
   bool use_grb = false, bool obj = false, int lazy = 0, int phint = -10, bool hint = false, bool use_bound = false, double bound = 0.0
) {
   if(obj && !use_grb) {
      const auto& [plus, minus] = gen_abs_obj_vars(model, name, phint, hint);
      model.addConstr(plus - minus == x, name).set(GRB_IntAttr_Lazy, lazy);
      return plus + minus;
   }
   return gen_abs_var(model, x, name, use_grb, obj, lazy, phint, hint, use_bound, bound);
}

GRBLinExpr gen_bin_w_expr(
   GRBModel& model, const GRBVar& b, const GRBLinExpr& a, const string& name,
   double coef,     int lazy = 0,    bool use_sos = false, bool use_bound = false, double bound = 0.0
) {
   if(coef == 0.0) {
      return GRBLinExpr();
   }
   GRBVar bw = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
   if(use_bound) {
      model.addConstr(bw <= coef * a + bound *      b,  format("{}_aub", name)).set(GRB_IntAttr_Lazy, lazy);
      model.addConstr(bw >= coef * a - bound *      b,  format("{}_alb", name)).set(GRB_IntAttr_Lazy, lazy);
      model.addConstr(bw <=            bound * (1 - b), format("{}_blb", name)).set(GRB_IntAttr_Lazy, lazy);
      model.addConstr(bw >=           -bound * (1 - b), format("{}_bub", name)).set(GRB_IntAttr_Lazy, lazy);
   }
   else {
      model.addGenConstrIndicator(b, 0, bw == coef * a, format("{}_off", name));
      if(use_sos) {
         array<GRBVar, 2> vars_b{b, bw}; array<double, 2> weights_b{0, 1};
         model.addSOS(vars_b.data(), weights_b.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(b, 1, bw == 0, format("{}_on",  name));
      }
   }
   return bw;
}

auto gen_ReLU_vars(
   GRBModel& model, const GRBLinExpr& z,  const string& name,
   int lazy = 0,    bool use_sos = false, bool use_grb = false, bool use_bound = false, double bound = 0.0
) {
   const auto& var = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
   const auto& on_0 = model.addVar(0, 1, 0, GRB_BINARY, format("{}_lb", name));
   if(!use_bound) {
      model.addGenConstrIndicator(on_0, 0, var == z, format("{}_off",  name));
      if(use_sos) {
         array<GRBVar, 2> vars_z{on_0, var}; array<double, 2> weights_z{0, 1};
         model.addSOS(vars_z.data(), weights_z.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(on_0, 1, var <= 0, format("{}_on",  name));
      }
   }
   else {
      model.addConstr(var >= z,                  format("{}_ubz",name)).set(GRB_IntAttr_Lazy, lazy);
      model.addConstr(var <= z + bound * on_0,   format("{}_lbz",name)).set(GRB_IntAttr_Lazy, lazy);
      model.addConstr(var <= bound * (1 - on_0), format("{}_lb0",name)).set(GRB_IntAttr_Lazy, lazy);
   }
   return make_tuple(var, on_0);
}

GRBLinExpr gen_ReLU_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,
   int lazy = 0,    bool use_grb = false, bool use_sos = false, bool use_bound = false, double bound = 0.0
) {
   if(!use_grb || use_bound) {
      const auto& [var, on_z] = gen_ReLU_vars(model, z, name, lazy, use_sos, use_grb, use_bound, bound);
      return var;
   }
   const auto& var = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
   const auto& new_z = gen_var(
      model, z, format("{}_in", name),
      -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, lazy
   );
   model.addGenConstrMax(var, &new_z, 1, 0, name);
   return var;
}

GRBLinExpr gen_Hardtanh_expr(
   GRBModel& model,      const GRBLinExpr& z, const string& name,
   const pair<double, double>& lim = {-1,1},  int lazy = 0,
   bool use_grb = false, bool use_sos = false, bool use_bound = false, double bound = 0.0
) {
   const auto& zl = gen_ReLU_expr(model, z - lim.first,  format("{}_z0", name), lazy, use_grb, use_sos, use_bound, bound);
   const auto& zu = gen_ReLU_expr(model, z - lim.second, format("{}_z1", name), lazy, use_grb, use_sos, use_bound, bound);
   return lim.first + zl - zu;
}

GRBLinExpr gen_Hardsigmoid_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,
   int lazy = 0,    bool use_grb = false, bool use_sos = false, bool use_bound = false, double bound = 0.0
) {
   const auto& new_z  = z / 6 + 0.5;
   const auto& z0 = gen_ReLU_expr(model, new_z,     format("{}_z0", name), lazy, use_grb, use_sos, use_bound, bound);
   const auto& z1 = gen_ReLU_expr(model, new_z - 1, format("{}_z1", name), lazy, use_grb, use_sos, use_bound, bound);
   return z0 - z1;
}

GRBLinExpr gen_ReLU6_expr(
   GRBModel& model, const GRBLinExpr& z, const string& name,
   int lazy = 0, bool use_grb = false, bool use_sos = false, bool use_bound = false, double bound = 0.0
) {
   const auto& z0 = gen_ReLU_expr(model, z,     format("{}_z0", name), lazy, use_grb, use_sos, use_bound, bound);
   const auto& z6 = gen_ReLU_expr(model, z - 6, format("{}_z6", name), lazy, use_grb, use_sos, use_bound, bound);
   return z0 - z6;
}

GRBLinExpr gen_Hardshrink_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,
   int lazy = 0,    bool use_sos = false, double lambda = 0.5, bool use_grb = false, bool use_bound = false, double bound = 0.0
) {
   const auto& [upper, on_upper0] = gen_ReLU_vars(model,   z - lambda,  name, lazy, use_sos, use_grb, use_bound, bound);
   const auto& [lower, on_lower0] = gen_ReLU_vars(model, -(z + lambda), name, lazy, use_sos, use_grb, use_bound, bound);
   return upper + lower + lambda * (1 - on_upper0) - lambda * (1 - on_lower0);
}

GRBLinExpr gen_Softshrink_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,
   int lazy = 0,    bool use_grb = false, bool use_sos = false, double lambda = 0.5, bool use_bound = false, double bound = 0.0
) {
   const auto& zp = gen_ReLU_expr(model,   z - lambda,  format("{}_plus",  name), lazy, use_grb, use_sos, use_bound, bound);
   const auto& zm = gen_ReLU_expr(model, -(z + lambda), format("{}_minus", name), lazy, use_grb, use_sos, use_bound, bound);
   return zp - zm;
}

GRBLinExpr gen_Threshold_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,
   int lazy = 0,    bool use_sos = false, double threshold = 0.5, double value = 0.5, bool use_grb = false, bool use_bound = false, double bound = 0.0
) {
   const auto& [zt, on_0] = gen_ReLU_vars(model, z - threshold,  name, lazy, use_sos, use_grb, use_bound, bound);
   return zt + value + (threshold - value) * (1 - on_0);
}

GRBLinExpr gen_LeakyReLU_expr(
   GRBModel& model,     const GRBLinExpr& z, const string& name,
   double reluc = 0.25, int lazy = 0,        bool use_grb = false, bool use_bound = false, double bound = 0.0
) {
   const auto& z_abs = gen_abs_expr(model, z, name, use_grb, false, lazy, -20, false, use_bound, bound);
   const auto& min_z0 = (z - z_abs) / 2;
   const auto& max_z0 = (z + z_abs) / 2;
   return max_z0 + reluc * min_z0;
}

GRBLinExpr gen_act_w_expr(const GRBLinExprRange auto& bw) {
   return accumulate(ranges::begin(bw) + 1, ranges::end(bw), -*ranges::begin(bw));
}

GRBLinExpr gen_w_expr(
   GRBModel& model, const GRBLinExprRange auto& b, int exp = 4,
   int lazy = 0,    const GRBLinExpr& tw = 0
) {
   vec<GRBLinExpr> exp_exprs;
   for(const auto& [l, b] : b | views::enumerate) {
      exp_exprs.emplace_back(exp2(exp - l) * (1 - b));
   }
   return accumulate(exp_exprs.begin() + 1, exp_exprs.end(), -exp_exprs[0]) + tw;
}

GRBVar gen_w_var(
   GRBModel& model, const GRBLinExprRange auto& b, const string& name,
   int exp = 4,     int lazy = 0,                  const GRBLinExpr& tw = 0
) {
   return gen_var(
      model, gen_w_expr(model, b, exp, lazy, tw),
      name, -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

auto gen_activation_exprs(
   GRBModel& model,      const string& type,                 const GRBLinExprRange auto& z,
   const string& name,   const RangeOf<double> auto& params, int lazy = 0, 
   bool use_grb = false, bool use_sos = false, bool use_bound = false, double bound = 0.0
) {
   vec<GRBLinExpr> a; string new_name;
   for(unsigned char c : type) {
      new_name.push_back(tolower(c));
   }
   new_name = format("{}_{}", name, new_name);
   if(type == "ReLU") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_ReLU_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, use_grb, use_sos, use_bound, bound
         ));
      }
   }
   else if(type == "ReLU6") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_ReLU6_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, use_grb, use_sos, use_bound, bound
         ));
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      vec<double> values(ranges::distance(z), 0.25);
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z, leakyreluc] : views::zip(views::iota(0), z, values)) {
         a.emplace_back(gen_LeakyReLU_expr(
            model, z,
            format("{}_{}", new_name, j),
            leakyreluc, lazy, use_grb, use_bound, bound
         ));
      }
   }
   else if(type  == "Hardtanh") {
      array<double, 2> values{-1, 1};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_Hardtanh_expr(
            model, z,
            format("{}_{}", new_name, j),
            {values[0], values[1]}, lazy, use_grb, use_sos, use_bound, bound
         ));
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_Hardsigmoid_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, use_grb, use_sos, use_bound, bound
         ));
      }
   }
   else if(type == "Hardshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_Hardshrink_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, use_sos, values[0], use_grb, use_bound, bound
         ));
      }
   }
   else if(type == "Softshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_Softshrink_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, use_grb, use_sos, values[0], use_bound, bound
         ));
      }
   }
   else if(type == "Threshold") {
      array<double, 2> values{0.5, 0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_Threshold_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, use_sos, values[0], values[1], use_grb, use_bound, bound
         ));
      }
   }
   else {
      ranges::move(z, back_inserter(a));
   } 
   return a;
}

GRBLinExpr gen_class_error_expr(
   GRBModel& model,                const GRBLinExprRange auto& y, const string& name,
   const RangeOf<double> auto& ty, int instances,                 const RangeOf<double> auto& cls_w,
   double tol0 = 0.000000000001,   double tol = 0.1,              int lazy = 0,
   bool use_grb = false,           bool use_square = false,       bool hint = true,
   double offpen = 10,             bool use_bound = false,        double bound = 0.0
) {
   vec<GRBLinExpr> exprs;
   int size = ranges::distance(ty);
   double tp = 1.0 / max<int>(2, size);
   auto bound_down = [&tp] (const auto& ty) {return ty > tp ? tp : 0;};
   auto bound_up   = [&tp] (const auto& ty) {return ty < tp ? tp : 1;};
   if(size == 1) {
      // Clasificación binaria usando sigmoide
      double ty_i    = *ranges::begin(ty);
      GRBLinExpr y_i = *ranges::begin(y);
      double cls_w_i = *ranges::begin(cls_w);
      cls_w_i = ty_i < 0.5 ? instances - cls_w_i : cls_w_i;
      double cls_weight = max(tol0, cls_w_i);
      double lb_ty = bound_down(ty_i);
      double ub_ty = bound_up  (ty_i);
      double lb_out = inverse_sigmoid(lb_ty, tol0);
      double ub_out = inverse_sigmoid(ub_ty, tol0);
      GRBVar lower = model.addVar(0, 1, 0, GRB_BINARY, format("{}_lower", name));
      GRBVar upper = model.addVar(0, 1, 0, GRB_BINARY, format("{}_upper", name));
      if(use_bound) {
            model.addConstr(y_i <= lb_out + bound * (1 - lower)).set(GRB_IntAttr_Lazy, lazy);
            model.addConstr(y_i >= lb_out - bound *      lower ).set(GRB_IntAttr_Lazy, lazy);
            model.addConstr(y_i <= ub_out + bound *      upper ).set(GRB_IntAttr_Lazy, lazy);
            model.addConstr(y_i >= ub_out - bound * (1 - upper)).set(GRB_IntAttr_Lazy, lazy);
      }
      else {
         model.addGenConstrIndicator(lower, 1, y_i <= lb_out, format("{}_loweron",  name));
         model.addGenConstrIndicator(lower, 0, y_i >= lb_out, format("{}_loweroff", name));
         model.addGenConstrIndicator(upper, 0, y_i <= ub_out, format("{}_upperoff", name));
         model.addGenConstrIndicator(upper, 1, y_i >= ub_out, format("{}_upperon",  name));
      }
      if(!use_square) {
         double lb = inverse_sigmoid(max(-tol + ty_i, bound_down(ty_i)), tol0);
         double ub = inverse_sigmoid(min( tol + ty_i, bound_up  (ty_i)), tol0);
         exprs.emplace_back((
            gen_abs_range_obj_expr(model, y_i, name, lb, ub, lazy, use_grb, 10, hint) +
               lower * exp(offpen) + upper * exp(offpen)
         ) / cls_weight);
      }
      else {
         double target = inverse_sigmoid(ty_i, tol0);
         exprs.emplace_back((
            gen_approx_square_obj_expr(model, y_i - target, name, lazy, use_grb, 10, hint) +
               lower * exp(offpen) + upper * exp(offpen)
         ) / cls_weight);
      }
   }
   else if(size > 1) {
      // Clasificación multiclase usando softmax
      GRBVar c = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_C",name));
      for(const auto& [cls, ty, y, cls_w] : views::zip(views::iota(0), ty, y, cls_w)) {
         const string& name_i = format("{}_{}", name, cls);
         const GRBLinExpr& expr_i(y - c);
         double cls_weight = max(tol0, cls_w);
         double lb_ty = bound_down(ty);
         double ub_ty = bound_up  (ty);
         double lb_out = log(max(lb_ty, tol0));
         double ub_out = log(max(ub_ty, tol0));
         GRBVar lower = model.addVar(0, 1, 0, GRB_BINARY, format("{}_lower", name_i));
         GRBVar upper = model.addVar(0, 1, 0, GRB_BINARY, format("{}_upper", name_i));
         if(use_bound) {
            model.addConstr(expr_i <= lb_out + bound * (1 - lower)).set(GRB_IntAttr_Lazy, lazy);
            model.addConstr(expr_i >= lb_out - bound *      lower ).set(GRB_IntAttr_Lazy, lazy);
            model.addConstr(expr_i <= ub_out + bound *      upper ).set(GRB_IntAttr_Lazy, lazy);
            model.addConstr(expr_i >= ub_out - bound * (1 - upper)).set(GRB_IntAttr_Lazy, lazy);
         }
         else {
            model.addGenConstrIndicator(lower, 1, expr_i <= lb_out, format("{}_loweron",  name_i));
            model.addGenConstrIndicator(lower, 0, expr_i >= lb_out, format("{}_loweroff", name_i));
            model.addGenConstrIndicator(upper, 0, expr_i <= ub_out, format("{}_upperoff", name_i));
            model.addGenConstrIndicator(upper, 1, expr_i >= ub_out, format("{}_upperon",  name_i));
         }
         if(!use_square) {
            double lb = log(max(max(-tol + ty, lb_ty), tol0));
            double ub = log(max(min( tol + ty, ub_ty), tol0));
            exprs.emplace_back((
               gen_abs_range_obj_expr(model, expr_i, name_i, lb, ub, lazy, use_grb, 10, hint) +
               lower * exp(offpen) + upper * exp(offpen)
            ) / cls_weight);
         }
         else {
            double target = log(max(ty, tol0));
            exprs.emplace_back((
               gen_approx_square_obj_expr(model, expr_i - target, name, lazy, use_grb, 10, hint) + 
               lower * exp(offpen) + upper * exp(offpen)
            ) / cls_weight);
         }
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_regression_error_expr(
   GRBModel& model, const GRBLinExprRange auto& y, const string& name,
   const RangeOf<double> auto& ty, double tol0 = 0.0001,
   double rtol = 0.0, int lazy = 0, bool use_grb = false, bool use_sqr = false,
   bool hint = true
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [i, y, ty] : views::zip(views::iota(0), y, ty)) {
      const string& name_i = format("{}_{}", name, i);
      const GRBLinExpr expr_i(y - ty);
      if(!use_sqr) {
         double ty0 = max(abs(ty), tol0);
         double lb = -rtol * ty0;
         double ub =  rtol * ty0;
         exprs.emplace_back(
            gen_abs_range_obj_expr(model, expr_i, name_i, lb, ub, lazy, use_grb, 10, hint)
         );
      }
      else {
         exprs.emplace_back(
            gen_approx_square_obj_expr(model, expr_i, name_i, lazy, use_grb, 10, hint)
         );
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l1w_expr(
   GRBModel& model, const MatrixRange<GRBVar> auto& w, const string& name,
   int lazy = 0, double tol = 0, bool use_grb = false, bool hint = true
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, w] : w | views::enumerate) {
      for(const auto& [j, w] : w | views::enumerate) {
         exprs.emplace_back(gen_abs_range_obj_expr(
            model, w, format("{}_{}_{}", name, i, j),
            -tol, tol, lazy, use_grb, -10, hint
         ));
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l2w_expr(
   GRBModel& model, const MatrixRange<GRBVar> auto& w,
   const string& name, int lazy = 0, bool use_grb = false, bool hint = true
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, w] : w | views::enumerate) {
      for(const auto& [j, w] : w | views::enumerate) {
         exprs.emplace_back(gen_approx_square_obj_expr(
            model, w, format("{}_{}_{}", name, i, j),
            lazy, use_grb, -10, hint
         ));
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l1a_expr(
   GRBModel& model, const GRBLinExprRange auto& a, const string& name,
   int lazy = 0, double tol = 0, bool use_grb = false, bool hint = true
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [j, a] : a | views::enumerate) {
      exprs.emplace_back(gen_abs_range_obj_expr(
         model, a, format("{}_{}", name, j),
         -tol, tol, lazy, use_grb, -10, hint
      ));
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l2a_expr(
   GRBModel& model,    const GRBLinExprRange auto& a,
   const string& name, int lazy = 0, bool use_grb = false,
   bool hint = true
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [j, a] : a | views::enumerate) {
      exprs.emplace_back(gen_approx_square_obj_expr(
         model, a, format("{}_{}", name, j),
         lazy, use_grb, -10, hint
      ));
   }
   return gen_mean_expr(exprs);
}


GRBLinExpr gen_wp_expr(
   GRBModel& model,       const MatrixRange<GRBVar> auto& w,
   const MatrixRange<optional<double>> auto& tw,
   const string& name,    int lazy = 0,         double tol = 0,
   double tol0 = 0.00001, bool use_grb = false, bool use_sqr = false,
   bool hint = true
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, w, tw] : views::zip(views::iota(0), w, tw)) {
      for(const auto& [j, w, tw] : views::zip(views::iota(0), w, tw)) {
         if(tw) {
            if(!use_sqr) {
               double tw0 = max(abs(*tw), tol0);
               double lb = -tol * tw0;
               double ub =  tol * tw0;
               exprs.emplace_back(gen_abs_range_obj_expr(
                  model, w - *tw, format("{}_{}_{}", name, i, j),
                  lb, ub, lazy, use_grb, 5, hint
               ));
            }
            else {
               exprs.emplace_back(gen_approx_square_obj_expr(
                  model, w - *tw, format("{}_{}_{}", name, i, j),
                  lazy, use_grb, 5, hint
               ));
            }
         }
      }
   }
   return gen_mean_expr(exprs);
}

auto gen_class_count(const Tensor3Range<double> auto& ty) {
   vec<vec<double>> cls_w;
   for(auto& ty : ty) {
      cls_w.emplace_back(vec<double>(ranges::distance(*ranges::begin(ty)), 0.0));
      for(const auto& ty : ty) {
         for(auto&& [ty, cls_w] : views::zip(ty, cls_w.back())) {
            cls_w += ty;
         }
      }
   }
   return cls_w;
}

auto gen_layers(
   GRBModel& model, int layers,               const RangeOf<int> auto& cap,
   const Tensor3Range<bool> auto& mask,       const Tensor3Range<int> auto& exp,
   const Tensor3Range<int> auto& bits,        const Tensor3Range<optional<double>> auto& tw,
   const RangeOf<optional<double>> auto& l1w, const RangeOf<optional<double>> auto& l2w,
   const optional<double>& wpen,              const Tensor3Range<bool> auto& fixed,
   int lazy = 1, bool relax_model = false,    bool use_start = false, bool use_grb = false,
   double offtol = 0.0,                       double tol0 = 0.000001, double rtol = 0.0,
   bool alt_model = false,                    bool use_sqr = false,
   bool no_l1 = false, bool no_l2 = false,    bool hint = true
) {
   GRBVar one_var  = model.addVar(1, 1, 0, GRB_BINARY, "one_singleton" );
   GRBVar zero_var = model.addVar(0, 0, 0, GRB_BINARY, "zero_singleton");
   if(hint) {
      set_hint(one_var,  1, 20); set_hint(zero_var, 0, 20);
   }
   if(use_start) {
      int starts = 2;
      if(relax_model && alt_model) {
         starts += 1;
      }
      else if(!relax_model && !alt_model) {
         starts += 1;
      }
      else if(relax_model && !alt_model) {
         starts += 2;
      }
      resetline_console();
      cout << "Starts = " << starts << "\n";
      model.set(GRB_IntAttr_NumStart, starts);
      for(int i = 0; i < starts; ++i) {
         set_start(model, zero_var, i, 0);
         set_start(model, one_var,  i, 1);
      }
   }
   ten4<GRBVar> b(layers); ten3<GRBVar> iw(layers);
   vec<GRBLinExpr> mask_exprs, l1w_exprs, l2w_exprs, wpen_exprs;
   for(    auto&& [k, b, tw, bits, mask, exp, fixed, l1w, l2w, iw, sizes] : views::zip(
      views::iota(0), b, tw, bits, mask, exp, fixed, l1w, l2w, iw, cap | views::adjacent<2>
   )) {
      const auto& [n, m] = sizes;
      b = ten3<GRBVar>(n + 1, mat<GRBVar>(m));
      iw = mat<GRBVar>(n + 1, vec<GRBVar>(m));
      mat<GRBVar>    w(n + 1, vec<GRBVar>(m));
      for(   auto&& [i, b, w, tw, bits, exp, mask, fixed, iw] : views::zip(views::iota(0), b, w, tw, bits, exp, mask, fixed, iw)) {
         for(auto&& [j, b, w, tw, bits, exp, mask, fixed, iw] : views::zip(views::iota(0), b, w, tw, bits, exp, mask, fixed, iw)) {
            if(!tw) {
               model.set(GRB_IntAttr_NumStart, 1);
            }
            const auto& [b_floor, b_ceil] = decompose_w(tw, bits + 1);
            b = vec<GRBVar>(bits + 1);
            if(fixed) {
               ranges::fill(b, one_var);
               iw = zero_var;
            }
            else if(!relax_model && !mask) {
               ranges::fill(b, one_var);
               iw = one_var;
            }
            else {
               for(auto&& [l, b, bf, bc] : views::zip(views::iota(0), b, b_floor, b_ceil)) {
                  b = model.addVar(0, 1, 0, GRB_BINARY, format("b_{}_{}_{}_{}", k, i, j, l));
                  if(use_start) {
                     set_start(model, b, 0, 1);
                     if(tw && relax_model && alt_model) {
                        set_start(model, b, 1, 1);
                        set_start(model, b, 2, 1);
                     }
                     else if(tw && !relax_model && alt_model) {
                        set_start(model, b, 1, 1);
                     }
                     else if(tw && !relax_model && !alt_model) {
                        set_start(model, b, 1, (1 - *bf) * mask);
                        set_start(model, b, 2, (1 - *bc) * mask);
                     }
                     else if(tw && relax_model && !alt_model) {
                        set_start(model, b, 1, (1 - *bf));
                        set_start(model, b, 2, (1 - *bc));
                        set_start(model, b, 3, (1 - *bf) * mask);
                        set_start(model, b, 4, (1 - *bc) * mask);
                     }
                  }
               }
               if(tw && alt_model) {
                  iw = model.addVar(0, 1, 0, GRB_BINARY, format("iw_{}_{}_{}", k, i, j));
                  if(use_start) {
                     set_start(model, iw, 0, 1);
                     if(relax_model) {
                        set_start(model, iw, 1, 0);
                        set_start(model, iw, 2, !mask);
                     }
                     else {
                        set_start(model, iw, 1, !mask);
                     }
                  }
               }
               else {
                  iw = one_var;
               }
            }
            w = gen_w_var(
               model, b, format("w_{}_{}_{}", k, i, j),
               exp, 0, tw.value_or(0.0) * (1 - iw)
            );
            if(relax_model && !mask) {
               if(!use_sqr) {
                  mask_exprs.emplace_back(gen_abs_range_obj_expr(
                     model, w, format("mask_{}_{}_{}", k, i, j),
                     -offtol, offtol, lazy, use_grb, 5, hint
                  ));
               }
               else {
                  mask_exprs.emplace_back(gen_approx_square_obj_expr(
                     model, w, format("mask_{}_{}_{}", k, i, j),
                     lazy, use_grb, 5, hint
                  ));
               }
            }
            else if(!relax_model && !mask) {
               w.set(GRB_DoubleAttr_LB, 0);
               w.set(GRB_DoubleAttr_UB, 0);
               if(hint) {
                  set_hint(w, 0, 10);
               }
            }
            if(use_start) {
               set_start(model, w, 0, 0);
               if(tw && relax_model && alt_model) {
                  set_start(model, w, 1, *tw);
                  set_start(model, w, 2, *tw * mask);
               }
               else if(tw && !relax_model && alt_model) {
                  set_start(model, w, 1, *tw * mask);
               }
               else if(tw && !relax_model && !alt_model) {
                  set_start(model, w, 1, calculate_w(b_floor, exp) * mask);
                  set_start(model, w, 2, calculate_w(b_ceil,  exp) * mask);
               }
               else if(tw && relax_model && !alt_model) {
                  set_start(model, w, 1, calculate_w(b_floor, exp));
                  set_start(model, w, 2, calculate_w(b_ceil,  exp));
                  set_start(model, w, 3, calculate_w(b_floor, exp) * mask);
                  set_start(model, w, 4, calculate_w(b_ceil,  exp) * mask);
               }
            } 
         }
      }
      if(wpen) {
         wpen_exprs.emplace_back(*wpen * gen_wp_expr(
            model, w, tw, format("wp_{}", k),
            lazy, rtol, tol0, use_grb, use_sqr, hint
         ));
      }
      if(!no_l1 && l1w && k + 1 < layers) {
         l1w_exprs.emplace_back(*l1w * gen_l1w_expr(
            model, w, format("l1w_{}", k),
            lazy, offtol, use_grb, hint
         ));
      }
      if(!no_l2 && l2w && k + 1 < layers) {
         l2w_exprs.emplace_back(*l2w * gen_l2w_expr(
            model, w, format("l2w_{}", k),
            lazy, use_grb, hint
         ));
      }
   }
   return make_tuple(b, mask_exprs, l1w_exprs, l2w_exprs, wpen_exprs, iw);
}

auto gen_dropout_exprs(
   GRBModel& model,    const GRBLinExprRange auto& a, const RangeOf<bool> auto& drop,
   const string& name, bool relax_model = false,      int lazy = 1,
   double dtol = 0.0,  bool use_grb = true,           bool use_sqr = false,
   bool hint = true
) {
   GRBLinExpr obj_expr; vec<GRBLinExpr> a_exprs(ranges::begin(a), ranges::end(a));
   if(relax_model) {
      vec<GRBLinExpr> drop_exprs;
      for(const auto& [i, a, drop] : views::zip(views::iota(0), a, drop)) {
         if(drop) {
            if(!use_sqr) {
               drop_exprs.emplace_back(gen_abs_range_obj_expr(
                  model, a, format("{}_{}", name, i),
                  -dtol, dtol, lazy, use_grb, 5, hint
               ));
            }
            else {
               drop_exprs.emplace_back(gen_approx_square_obj_expr(
                  model, a, format("{}_{}", name, i), lazy, use_grb, 5, hint
               ));
            }
         }
      }
      obj_expr = gen_mean_expr(drop_exprs);
   }
   else {
      for(auto&& [a_expr, drop] : views::zip(a_exprs, drop)) {
         if(drop) {
            a_expr = 0;
         }
      }
   }
   return make_tuple(a_exprs, obj_expr);
}

auto gen_connection_dropout_exprs(
   GRBModel& model,                    const GRBLinExprRange auto& a,
   const Tensor3Range<GRBVar> auto& b, int n, int m,                     const MatrixRange<bool> auto& cdrop,
   const MatrixRange<bool> auto& mask, const MatrixRange<int> auto& exp, const MatrixRange<optional<double>> auto& tw,
   const MatrixRange<GRBVar> auto& iw, const string& name,               bool relax_model = false,
   int lazy = 1,                       double dtol = 0.0,                bool use_grb = true,
   bool alt_model = false,             bool use_sqr = false,             bool hint = true,
   bool use_sos = false,               bool use_bound = false,           double bound = 0.0
) {
   GRBLinExpr obj_expr; mat<GRBLinExpr> aw_exprs(m, vec<GRBLinExpr>(n + 1));
   if(relax_model) {
      vec<GRBLinExpr> drop_exprs;
      for(const auto& [i, b, a, exp, cdrop, mask, tw, iw] : views::zip(views::iota(0), b, a, exp, cdrop, mask, tw, iw)) {
         for(const auto& [j, b, exp, cdrop, mask, tw, iw] : views::zip(views::iota(0), b,    exp, cdrop, mask, tw, iw)) {
            if(tw && alt_model) {
               aw_exprs[j][i] += gen_bin_w_expr(
                  model, iw, a, format("{}_{}_{}_w", name, i, j), *tw, lazy, use_sos, use_bound, bound
               );
            }
            vec<GRBLinExpr> bin_w;
            int bits = ranges::distance(b);
            for(const auto& [l, b] : b | views::enumerate) {
               bin_w.emplace_back(exp2(exp - bits) * gen_bin_w_expr(
                  model, b, a, format("{}_{}_{}_{}", name, i, j, l), exp2(bits - l), lazy, use_sos, use_bound, bound
               ));
            }
            aw_exprs[j][i] += gen_act_w_expr(bin_w);
            if(cdrop) {
               if(!use_sqr) {
                  drop_exprs.emplace_back(gen_abs_range_obj_expr(
                     model, aw_exprs[j][i], format("{}_{}_{}_condrop", name, i, j),
                     -dtol, dtol, lazy, use_grb, 5, hint
                  ));
               }
               else {
                  drop_exprs.emplace_back(gen_approx_square_obj_expr(
                     model, aw_exprs[j][i], format("{}_{}_{}_condrop", name, i, j),
                     lazy, use_grb, 5, hint
                  ));
               }
            }
         }
      }
      obj_expr = gen_mean_expr(drop_exprs);
   }
   else {
      for(const auto& [i, b, a, exp, cdrop, mask, tw, iw] : views::zip(views::iota(0), b, a, exp, cdrop, mask, tw, iw)) {
         for(const auto& [j, b, exp, cdrop, mask, tw, iw] : views::zip(views::iota(0), b,    exp, cdrop, mask, tw, iw)) {
            if(mask && !cdrop) {
               if(tw && alt_model) {
                  aw_exprs[j][i] += gen_bin_w_expr(
                     model, iw, a, format("{}_{}_{}_w", name, i, j), *tw, lazy, use_sos, use_bound, bound
                  );
               }
               vec<GRBLinExpr> bin_w;
               for(const auto& [l, b] : b | views::enumerate) {
                  bin_w.emplace_back(gen_bin_w_expr(
                     model, b, a, format("{}_{}_{}_{}", name, i, j, l), exp2(exp - l), lazy, use_sos, use_bound, bound
                  ));
               }
               aw_exprs[j][i] += gen_act_w_expr(bin_w);
            }
         }
      }
   }
   const auto& z = aw_exprs | views::transform(gen_sum_expr<decltype(aw_exprs)::value_type>) | views::common;
   return make_tuple(vec<GRBLinExpr>(z.begin(), z.end()), obj_expr);
}

auto get_model(
   const GRBEnv& environment,          int instances, int layers,                      const RangeOf<int> auto& cap,
   const RangeOf<string> auto& af,     const Tensor3Range<bool> auto& mask,            const Tensor3Range<int> auto& exp,
   const Tensor3Range<int> auto& bits, const Tensor3Range<optional<double>> auto& tw,  const RangeOf<double> auto& bias,
   const MatrixRange<double> auto& params,
   const MatrixRange<bool> auto& drop, const RangeOf<optional<double>> auto& l1a,      const RangeOf<optional<double>> auto& l1w,
   const RangeOf<optional<double>> auto& l2a,                                          const RangeOf<optional<double>> auto& l2w,
   const optional<double>& wpen,       const Tensor3Range<bool> auto& fixed,           const Tensor3Range<bool> auto& cdrop,
   const MatrixRange<double> auto& fx, const Tensor3Range<double> auto& reg_ty,        const Tensor3Range<double> auto& class_ty,
   double tol0 = 0.000000000001,       double relax_frac = 0.0, double rtol = 0.1,     int lazy = 1, bool relax_model = false,
   double err_prio = 1.0,              bool use_start = false, bool use_grb = false,   bool use_sos = false,
   double offtol = 0.0,                bool alt_model = false, bool use_sqr = false,   bool no_l1 = false, bool no_l2 = false,
   bool hint = true,                   double offpen = 10,     bool use_bound = false, double bound = 0.0
) {
   GRBModel model(environment);
   resetline_console();
   cout << "Processing layers variables..." << flush;
   const auto& [b, mask_exprs, l1w_exprs, l2w_exprs, wpen_exprs, iw] = gen_layers(
      model, layers, cap, mask, exp, bits, tw, l1w, l2w, wpen, fixed, lazy, relax_model, use_start,
      use_grb, offtol, tol0, rtol, alt_model, use_sqr, no_l1, no_l2, hint
   );
   vec<GRBLinExpr> drop_exprs, cdrop_exprs, l1a_exprs, l2a_exprs, target_exprs;
   const auto& cls_w = gen_class_count(class_ty);
   int relax_size = relax_frac * instances;
   for(const auto& [t, fx] : fx | views::enumerate) {
      resetline_console();
      cout << "Processing case: " << t << "...\n" << flush;
      vec<GRBLinExpr> a(ranges::begin(fx), ranges::end(fx)), l1a_exprs_i, l2a_exprs_i, target_exprs_i;
      for(const auto& [k, b, af, exp, drop, params, cdrop, l1a, l2a, bias, mask, tw, iw, sizes] :
      views::zip(
         views::iota (0), b, af, exp, drop, params, cdrop, l1a, l2a, bias, mask, tw, iw, cap | views::adjacent<2>
      )) {
         resetline_console();
         cout << "Processing layer: " << k << "..." << flush;
         const auto& [n, m] = sizes;
         a.emplace_back(bias);
         const auto& [drop_a, drop_expr] = gen_dropout_exprs(
            model, a, drop, format("drop_{}_{}", t, k),
            relax_model, lazy, offtol, use_grb, use_sqr, hint
         );
         drop_exprs.emplace_back(drop_expr);
         const auto& [z, cdrop_expr] = gen_connection_dropout_exprs(
            model, drop_a, b, n, m, cdrop, mask, exp, tw, iw,
            format("bw_{}_{}", t, k), relax_model, lazy,
            offtol, use_grb, alt_model, use_sqr, hint, use_sos, use_bound, bound
         );
         cdrop_exprs.emplace_back(cdrop_expr);
         a = gen_activation_exprs(model, af, z, format("act_{}_{}", t, k), params, lazy, use_grb, use_sos, use_bound, bound);
         if(!no_l1 && l1a && k + 1 < layers) {
            l1a_exprs_i.emplace_back(*l1a * gen_l1a_expr(model, a, format("l1a_{}_{}", t, k), lazy, offtol, use_grb, hint));
         }
         if(!no_l2 && l2a && k + 1 < layers) {
            l2a_exprs_i.emplace_back(*l2a * gen_l2a_expr(model, a, format("l2a_{}_{}", t, k), lazy, use_grb, hint));
         }
      }
      int asize = 0;
      for(const auto& [ti, ty, cls_w] : views::zip(
         views::iota  (0), class_ty, cls_w
      )) {
         const auto& ty_i = *(ranges::begin(ty) + t);
         int size = ranges::distance(ty_i);
         if(size > 0) {
            vec<GRBLinExpr> y;
            ranges::move(a | views::drop(asize) | views::take(size), back_inserter(y));
            target_exprs_i.emplace_back(gen_class_error_expr(
               model, y, format("cls_{}_{}", t, ti), ty_i, instances, cls_w,
               tol0, t < relax_size ? rtol : 0.0, lazy, use_grb, use_sqr, hint, offpen,
               use_bound, bound
            ));
            asize += size;
         }
      }
      for(const auto& [ti, ty] : reg_ty | views::enumerate) {
         const auto& ty_i = *(ranges::begin(ty) + t);
         int size = ranges::distance(ty_i);
         if (size > 0) {
            vec<GRBLinExpr> y;
            ranges::move(a | views::drop(asize) | views::take(size), back_inserter(y));
            target_exprs_i.emplace_back(gen_regression_error_expr(
               model, y, format("reg_{}_{}", t, ti), ty_i, tol0,
               t < relax_size ? rtol : 0.0, lazy, use_grb, use_sqr, hint
            ));
            asize += size;
         }
      }
      if(l2a_exprs_i.size() > 0) {
         l2a_exprs.emplace_back(gen_mean_expr(l2a_exprs_i));
      }
      if(l1a_exprs_i.size() > 0) {
         l1a_exprs.emplace_back(gen_mean_expr(l1a_exprs_i));
      }
      if(target_exprs_i.size() > 0) {
         target_exprs.emplace_back(gen_mean_expr(target_exprs_i));
      }
      resetline_console();
      cursorup_console(1);
   }
   resetline_console();
   cursorup_console(1);
   resetline_console();
   model.setObjective(
      instances * layers * (
         err_prio * gen_mean_expr(target_exprs) +
         gen_mean_expr(l1a_exprs) + gen_mean_expr(l1w_exprs) +
         gen_mean_expr(l2a_exprs) + gen_mean_expr(l2w_exprs) +
         gen_mean_expr(wpen_exprs) + gen_mean_expr(drop_exprs) +
         gen_mean_expr(cdrop_exprs) + gen_mean_expr(mask_exprs)
      ),
      GRB_MINIMIZE
   );
   return model;
}

/// 
/// Funciones relacionadas a main o lectura de datos
/// 

template<typename G>
auto gen_dropout_func(G&& generator, const optional<double>& dropout) {
   return [&generator, &dropout] () {
      return dropout.transform([&generator] (double dropout) {
         return bernoulli_distribution(dropout)(generator);
      }).value_or(false);
   };
}

template<typename G>
auto generate_dropouts(
   const RangeOf<int> auto& cap, const RangeOf<optional<double>> auto& dropout,
   const RangeOf<optional<double>> auto& connection_dropout, G&& generator, bool no_dropouts = false
) {
   vec<vec<bool>> dropout_bool  (ranges::distance(cap) - 1);
   vec<mat<bool>> c_dropout_bool(ranges::distance(cap) - 1);
   for(auto&& [k, db, pd, cdb, cpd, sizes] : views::zip(
      views::iota(0), dropout_bool, dropout, c_dropout_bool, connection_dropout, cap | views::adjacent<2>
   )) {
      const auto& [n, m] = sizes;
      db = vec<bool>(n + 1, false);
      cdb = mat<bool>(n + 1);
      if(!no_dropouts) {
         ranges::generate(db, gen_dropout_func(generator, pd));
      }
      for(auto& cdb : cdb) {
         cdb = vec<bool>(m, false);
         if(!no_dropouts) {
            ranges::generate(cdb, gen_dropout_func(generator, cpd));
         }
      }
   }
   return make_tuple(dropout_bool, c_dropout_bool);
}

template<typename T>
auto read_matrix_from_csv(istream& input, bool ignore_header = false, bool ignore_index = false) {
   mat<T> matrix; string line;
   if(ignore_header) {
      getline(input, line);
   }
   while(getline(input, line)) {
      ranges::replace(line, ',', ' ');
      vec<T> vector; stringstream line_stream(line);
      ranges::move(
         views::istream<T>(line_stream) | views::drop(ignore_index),
         back_inserter(vector)
      );
      matrix.emplace_back(vector);
   }
   return matrix;
}

template<typename T>
auto read_matrix_from_csv(istream&& input, bool ignore_header = false, bool ignore_index = false) {
   return read_matrix_from_csv<T>(input, ignore_header, ignore_index);
}

template<typename T>
auto read_list_from_csv(istream& input, bool ignore_index = false) {
   string line; vec<vec<int>> dim_list; vec<vec<T>> data_list;
   getline(input, line);
   ranges::replace(line, ',', ' ');
   stringstream line_stream(line);
   int max_dim = ranges::distance(
      views::istream<string>(line_stream) | views::drop(ignore_index) |
      views::take_while([] (const string& str) {
         return str.starts_with("d_");
      })
   );
   while(getline(input, line)) {
      ranges::replace(line, ',', ' ');
      stringstream line_stream(line); vec<int> dim;
      ranges::move(
         views::istream<int>(line_stream) | views::drop(ignore_index) |
         views::take(max_dim), back_inserter(dim)
      );
      vec<T> data;
      ranges::move(
         views::istream<T>(line_stream),
         back_inserter(data)
      );
      dim_list.emplace_back(dim);
      data_list.emplace_back(data);
   }
   return make_tuple(dim_list, data_list);
}

template<typename T>
auto read_list_from_csv(istream&& input, bool ignore_index = false) {
   return read_list_from_csv<T>(input, ignore_index);
}

auto read_arch(istream& input, bool ignore_index = false, bool ignore_header = true) {
   string line, word; vec<optional<double>> Drop, L1w, L1a, L2w, L2a, c_drop;
   vec<int> C; vec<string> AF; vec<double> bias;
   if(ignore_header) {
      getline(input, line);
   }
   while(getline(input, line)) {
      stringstream line_stream(line);
      if(ignore_index) {
         getline(line_stream, word, ',');
      }
      int k; optional<string> af;
      optional<double> l1a, l1w, l2a, l2w, drop, b, cd;
      getline(line_stream, word, ','); stringstream(word) >> k;
      getline(line_stream, word, ','); stringstream(word) >> af;
      getline(line_stream, word, ','); stringstream(word) >> b;
      getline(line_stream, word, ','); stringstream(word) >> drop;
      getline(line_stream, word, ','); stringstream(word) >> cd;
      getline(line_stream, word, ','); stringstream(word) >> l1w;
      getline(line_stream, word, ','); stringstream(word) >> l1a;
      getline(line_stream, word, ','); stringstream(word) >> l2w;
      getline(line_stream, word, ','); stringstream(word) >> l2a;
      C.emplace_back(k);
      AF.emplace_back(af.value_or("None"));
      bias.emplace_back(b.value_or(1.0));
      Drop.emplace_back(drop);
      c_drop.emplace_back(cd);
      L1w.emplace_back(l1w);
      L1a.emplace_back(l1a);
      L2w.emplace_back(l2w);
      L2a.emplace_back(l2a);
   }
   return make_tuple(C, AF, bias, Drop, c_drop, L1w, L1a, L2w, L2a);
}

auto read_arch(istream&& input, bool ignore_index = false, bool ignore_header = true) {
   return read_arch(input, ignore_index, ignore_header);
}

template<typename R, typename T>
auto get_layers_matrix(const MatrixRange<T> auto& data, const RangeOf<int> auto& cap) {
   vec<mat<R>> layers_data;
   for(const auto& [sizes, data] : views::zip(cap | views::adjacent<2>, data)) {
      const auto& [n, m] = sizes;
      mat<R> layer_data(n + 1);
      for(int i = 0; i <= n; ++i) {
         ranges::copy(
            data | views::drop(i * m) | views::take(m),
            back_inserter(layer_data[i])
         );
      }
      layers_data.emplace_back(layer_data);
   }
   return layers_data;
}

template<typename T>
auto clamp_layers_matrix(const Tensor3Range<T> auto& layers_matrix, T min_value, T max_value) {
   ten3<T> result(ranges::distance(layers_matrix));
   for(auto&& [a, b] : views::zip(result, layers_matrix)) {
      a = mat<T>(ranges::distance(b));
      for(auto&& [a, b] : views::zip(a, b)) {
         a = vec<T>(ranges::distance(b));
         for(auto&& [a, b] : views::zip(a, b)) {
            a = min(max_value, max(min_value, b));
         }
      }
   }
   return result;
}

variant< unordered_map< string, vec<string> >, string > process_opts(
   int argc, const char* argv[]
) {
   const static unordered_map< string, int > opts = {
      {"load_path",         1}, {"load_name",         1}, {"save_path",         1}, {"save_name",         1},
      {"no_log_to_console", 0}, {"no_sols",           0}, {"no_log",            0}, {"no_json",           0},
      {"no_sol",            0}, {"no_mst",            0}, {"no_ilp",            0}, {"no_lp",             0},
      {"no_opti",           0}, {"exp",               0}, {"mask",              0}, {"params",            0},
      {"fixed",             0}, {"bits",              0}, {"no_l1",             0}, {"no_l2",             0},
      {"no_drop",           0}, {"tol0",              1}, {"reltol",            1}, {"relax_frac",        1},
      {"err_prio",          1}, {"relax_model",       0}, {"init",              0}, {"start",             0},
      {"grb_con",           0}, {"alt_model",         0}, {"sos",               0}, {"square",            0},
      {"min_bits",          1}, {"max_bits",          1}, {"min_exp",           1}, {"max_exp",           1},
      {"lazy",              1}, {"offtol",            1}, {"wpen",              1}, {"seed",              1},
      {"no_hint",           0}, {"no_index",          0}, {"no_header",         0}, {"offpen",            1},
      {"samples",           1}, {"no_shuffle",        0}, {"bound",             1}
   };
   unordered_map< string, vec<string> > processed_opts;
   int argi = 1;
   while(argi < argc) {
      if(string arg = argv[argi]; !arg.starts_with("--")) {
         return format("Error: {} no es una opcion", arg);
      }
      else if(string arg_name = arg.substr(2); !opts.contains(arg_name) && !arg_name.starts_with("GRB")) {
         return format("Error: No existe la opcion {}", arg_name);
      }
      else if(int max_argi = argi + (arg_name.starts_with("GRB") ? 1 : opts.at(arg_name)); max_argi >= argc) {
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

void process_env(GRBEnv& env, const unordered_map< string, vec<string> >& processed_opts) {
   const static unordered_map< string, GRB_IntParam > grb_int_params = {
      {"SolutionLimit",    GRB_IntParam_SolutionLimit      }, {"Method",             GRB_IntParam_Method             },
      {"ConcurrentMethod", GRB_IntParam_ConcurrentMethod   }, {"ScaleFlag",          GRB_IntParam_ScaleFlag          },
      {"SimplexPricing",   GRB_IntParam_SimplexPricing     }, {"Quad",               GRB_IntParam_Quad               },
      {"NormAdjust",       GRB_IntParam_NormAdjust         }, {"Sifting",            GRB_IntParam_Sifting            },
      {"SiftMethod",       GRB_IntParam_SiftMethod         }, {"NetworkAlg",         GRB_IntParam_NetworkAlg         },
      {"LPWarmStart",      GRB_IntParam_LPWarmStart        }, {"SubMIPNodes",        GRB_IntParam_SubMIPNodes        },
      {"VarBranch",        GRB_IntParam_VarBranch          }, {"Cuts",               GRB_IntParam_Cuts               },
      {"CliqueCuts",       GRB_IntParam_CliqueCuts         }, {"CoverCuts",          GRB_IntParam_CoverCuts          },
      {"FlowCoverCuts",    GRB_IntParam_FlowCoverCuts      }, {"FlowPathCuts",       GRB_IntParam_FlowPathCuts       },
      {"GUBCoverCuts",     GRB_IntParam_GUBCoverCuts       }, {"ImpliedCuts",        GRB_IntParam_ImpliedCuts        },
      /*{"DualImpliedCuts",  GRB_IntParam_DualImpliedCuts    },*/ {"ProjImpliedCuts",    GRB_IntParam_ProjImpliedCuts    },
      {"MIPSepCuts",       GRB_IntParam_MIPSepCuts         }, {"MIRCuts",            GRB_IntParam_MIRCuts            },
      {"StrongCGCuts",     GRB_IntParam_StrongCGCuts       }, {"ModKCuts",           GRB_IntParam_ModKCuts           },
      {"ZeroHalfCuts",     GRB_IntParam_ZeroHalfCuts       }, {"NetworkCuts",        GRB_IntParam_NetworkCuts        },
      {"SubMIPCuts",       GRB_IntParam_SubMIPCuts         }, {"InfProofCuts",       GRB_IntParam_InfProofCuts       },
      {"RelaxLiftCuts",    GRB_IntParam_RelaxLiftCuts      }, {"RLTCuts",            GRB_IntParam_RLTCuts            },
      {"BQPCuts",          GRB_IntParam_BQPCuts            }, {"PSDCuts",            GRB_IntParam_PSDCuts            },
      {"LiftProjectCuts",  GRB_IntParam_LiftProjectCuts    }, {"MixingCuts",         GRB_IntParam_MixingCuts         },
      {"CutAggPasses",     GRB_IntParam_CutAggPasses       }, {"CutPasses",          GRB_IntParam_CutPasses          },
      {"GomoryPasses",     GRB_IntParam_GomoryPasses       }, {"NodeMethod",         GRB_IntParam_NodeMethod         },
      {"Presolve",         GRB_IntParam_Presolve           }, {"Aggregate",          GRB_IntParam_Aggregate          },
      {"IISMethod",        GRB_IntParam_IISMethod          }, {"PreCrush",           GRB_IntParam_PreCrush           },
      {"PreDepRow",        GRB_IntParam_PreDepRow          }, {"PrePasses",          GRB_IntParam_PrePasses          },
      {"DisplayInterval",  GRB_IntParam_DisplayInterval    }, {"OutputFlag",         GRB_IntParam_OutputFlag         },
      {"Threads",          GRB_IntParam_Threads            }, {"BarIterLimit",       GRB_IntParam_BarIterLimit       },
      {"Crossover",        GRB_IntParam_Crossover          }, {"CrossoverBasis",     GRB_IntParam_CrossoverBasis     },
      {"BarCorrectors",    GRB_IntParam_BarCorrectors      }, {"BarOrder",           GRB_IntParam_BarOrder           },
      {"PumpPasses",       GRB_IntParam_PumpPasses         }, {"RINS",               GRB_IntParam_RINS               },
      {"Symmetry",         GRB_IntParam_Symmetry           }, {"MIPFocus",           GRB_IntParam_MIPFocus           },
      {"NumericFocus",     GRB_IntParam_NumericFocus       }, {"AggFill",            GRB_IntParam_AggFill            },
      {"PreDual",          GRB_IntParam_PreDual            }, {"SolutionNumber",     GRB_IntParam_SolutionNumber     },
      {"MinRelNodes",      GRB_IntParam_MinRelNodes        }, {"ZeroObjNodes",       GRB_IntParam_ZeroObjNodes       },
      {"BranchDir",        GRB_IntParam_BranchDir          }, {"DegenMoves",         GRB_IntParam_DegenMoves         },
      {"InfUnbdInfo",      GRB_IntParam_InfUnbdInfo        }, {"DualReductions",     GRB_IntParam_DualReductions     },
      {"BarHomogeneous",   GRB_IntParam_BarHomogeneous     }, {"PreQLinearize",      GRB_IntParam_PreQLinearize      },
      {"MIQCPMethod",      GRB_IntParam_MIQCPMethod        }, {"NonConvex",          GRB_IntParam_NonConvex          },
      {"QCPDual",          GRB_IntParam_QCPDual            }, {"LogToConsole",       GRB_IntParam_LogToConsole       },
      {"PreSOS1Encoding",  GRB_IntParam_PreSOS1Encoding    }, {"PreSOS2Encoding",    GRB_IntParam_PreSOS2Encoding    },
      {"PreSparsify",      GRB_IntParam_PreSparsify        }, {"PreMIQCPForm",       GRB_IntParam_PreMIQCPForm       },
      {"Seed",             GRB_IntParam_Seed               }, {"ConcurrentMIP",      GRB_IntParam_ConcurrentMIP      },
      {"ConcurrentJobs",   GRB_IntParam_ConcurrentJobs     }, {"DistributedMIPJobs", GRB_IntParam_DistributedMIPJobs },
      {"LazyConstraints",  GRB_IntParam_LazyConstraints    }, {"TuneResults",        GRB_IntParam_TuneResults        },
      {"TuneTrials",       GRB_IntParam_TuneTrials         }, {"TuneOutput",         GRB_IntParam_TuneOutput         },
      {"TuneJobs",         GRB_IntParam_TuneJobs           }, {"TuneCriterion",      GRB_IntParam_TuneCriterion      },
      {"TuneMetric",       GRB_IntParam_TuneMetric         }, {"TuneDynamicJobs",    GRB_IntParam_TuneDynamicJobs    },
      {"Disconnected",     GRB_IntParam_Disconnected       }, {"UpdateMode",         GRB_IntParam_UpdateMode         },
      {"Record",           GRB_IntParam_Record             }, {"ObjNumber",          GRB_IntParam_ObjNumber          },
      {"MultiObjMethod",   GRB_IntParam_MultiObjMethod     }, {"MultiObjPre",        GRB_IntParam_MultiObjPre        },
      {"PoolSolutions",    GRB_IntParam_PoolSolutions      }, {"PoolSearchMode",     GRB_IntParam_PoolSearchMode     },
      {"ScenarioNumber",   GRB_IntParam_ScenarioNumber     }, {"StartNumber",        GRB_IntParam_StartNumber        },
      {"StartNodeLimit",   GRB_IntParam_StartNodeLimit     }, {"IgnoreNames",        GRB_IntParam_IgnoreNames        },
      {"PartitionPlace",   GRB_IntParam_PartitionPlace     }, {"CSPriority",         GRB_IntParam_CSPriority         },
      {"CSTLSInsecure",    GRB_IntParam_CSTLSInsecure      }, {"CSIdleTimeout",      GRB_IntParam_CSIdleTimeout      },
      {"ServerTimeout",    GRB_IntParam_ServerTimeout      }, {"TSPort",             GRB_IntParam_TSPort             },
      {"JSONSolDetail",    GRB_IntParam_JSONSolDetail      }, {"CSBatchMode",        GRB_IntParam_CSBatchMode        },
      {"FuncPieces",       GRB_IntParam_FuncPieces         }, {"CSClientLog",        GRB_IntParam_CSClientLog        },
      {"IntegralityFocus", GRB_IntParam_IntegralityFocus   }, {"NLPHeur",            GRB_IntParam_NLPHeur            },
      {"WLSTokenDuration", GRB_IntParam_WLSTokenDuration   }, {"LicenseID",          GRB_IntParam_LicenseID          },
      {"OBBT",             GRB_IntParam_OBBT               }, {"FuncNonlinear",      GRB_IntParam_FuncNonlinear      },
      {"SolutionTarget",   GRB_IntParam_SolutionTarget     }/*, {"ThreadLimit",        GRB_IntParam_ThreadLimit        }*/
   };
   const static unordered_map< string, GRB_DoubleParam > grb_double_params = {
      {"Cutoff",           GRB_DoubleParam_Cutoff          }, {"IterationLimit", GRB_DoubleParam_IterationLimit   },
      {"MemLimit",         GRB_DoubleParam_MemLimit        }, {"SoftMemLimit",      GRB_DoubleParam_SoftMemLimit     },
      {"NodeLimit",        GRB_DoubleParam_NodeLimit       }, {"TimeLimit",         GRB_DoubleParam_TimeLimit        },
      {"WorkLimit",        GRB_DoubleParam_WorkLimit       }, {"FeasibilityTol",    GRB_DoubleParam_FeasibilityTol   },
      {"IntFeasTol",       GRB_DoubleParam_IntFeasTol      }, {"MarkowitzTol",      GRB_DoubleParam_MarkowitzTol     },
      {"MIPGap",           GRB_DoubleParam_MIPGap          }, {"MIPGapAbs",         GRB_DoubleParam_MIPGapAbs        },
      {"OptimalityTol",    GRB_DoubleParam_OptimalityTol   }, {"PerturbValue",      GRB_DoubleParam_PerturbValue     },
      {"Heuristics",       GRB_DoubleParam_Heuristics      }, {"ObjScale",          GRB_DoubleParam_ObjScale         },
      {"NodefileStart",    GRB_DoubleParam_NodefileStart   }, {"BarConvTol",        GRB_DoubleParam_BarConvTol       },
      {"BarQCPConvTol",    GRB_DoubleParam_BarQCPConvTol   }, {"PSDTol",            GRB_DoubleParam_PSDTol           },
      {"ImproveStartGap",  GRB_DoubleParam_ImproveStartGap }, {"ImproveStartNodes", GRB_DoubleParam_ImproveStartNodes},
      {"ImproveStartTime", GRB_DoubleParam_ImproveStartTime}, {"FeasRelaxBigM",     GRB_DoubleParam_FeasRelaxBigM    },
      {"TuneTimeLimit",    GRB_DoubleParam_TuneTimeLimit   }, {"TuneCleanup",       GRB_DoubleParam_TuneCleanup      },
      {"TuneTargetMIPGap", GRB_DoubleParam_TuneTargetMIPGap}, {"TuneTargetTime",    GRB_DoubleParam_TuneTargetTime   },
      {"PreSOS1BigM",      GRB_DoubleParam_PreSOS1BigM     }, {"PreSOS2BigM",       GRB_DoubleParam_PreSOS2BigM      },
      {"PoolGap",          GRB_DoubleParam_PoolGap         }, {"PoolGapAbs",        GRB_DoubleParam_PoolGapAbs       },
      {"BestObjStop",      GRB_DoubleParam_BestObjStop     }, {"BestBdStop",        GRB_DoubleParam_BestBdStop       },
      {"CSQueueTimeout",   GRB_DoubleParam_CSQueueTimeout  }, {"FuncPieceError",    GRB_DoubleParam_FuncPieceError   },
      {"FuncPieceLength",  GRB_DoubleParam_FuncPieceLength }, {"FuncPieceRatio",    GRB_DoubleParam_FuncPieceRatio   },
      {"FuncMaxVal",       GRB_DoubleParam_FuncMaxVal      }, {"NoRelHeurTime",     GRB_DoubleParam_NoRelHeurTime    },
      {"NoRelHeurWork",    GRB_DoubleParam_NoRelHeurWork   }, {"WLSTokenRefresh",   GRB_DoubleParam_WLSTokenRefresh  }
   };
   const static unordered_map< string, GRB_StringParam > grb_string_params = {
      {"LogFile",          GRB_StringParam_LogFile         }, {"NodefileDir",       GRB_StringParam_NodefileDir      },
      {"ResultFile",       GRB_StringParam_ResultFile      }, {"WorkerPool",        GRB_StringParam_WorkerPool       },
      {"WorkerPassword",   GRB_StringParam_WorkerPassword  }, {"ComputeServer",     GRB_StringParam_ComputeServer    },
      {"ServerPassword",   GRB_StringParam_ServerPassword  }, {"CSRouter",          GRB_StringParam_CSRouter         },
      {"CSGroup",          GRB_StringParam_CSGroup         }, {"TokenServer",       GRB_StringParam_TokenServer      },
      {"CloudAccessID",    GRB_StringParam_CloudAccessID   }, {"CloudSecretKey",    GRB_StringParam_CloudSecretKey   },
      {"CloudPool",        GRB_StringParam_CloudPool       }, {"CloudHost",         GRB_StringParam_CloudHost        },
      {"JobID",            GRB_StringParam_JobID           }, {"CSManager",         GRB_StringParam_CSManager        },
      {"CSAuthToken",      GRB_StringParam_CSAuthToken     }, {"CSAPIAccessID",     GRB_StringParam_CSAPIAccessID    },
      {"CSAPISecret",      GRB_StringParam_CSAPISecret     }, {"UserName",          GRB_StringParam_UserName         },
      {"CSAppName",        GRB_StringParam_CSAppName       }, {"SolFiles",          GRB_StringParam_SolFiles         },
      {"WLSAccessID",      GRB_StringParam_WLSAccessID     }, {"WLSSecret",         GRB_StringParam_WLSSecret        },
      {"WLSToken",         GRB_StringParam_WLSToken        }, /*{"WLSProxy",          GRB_StringParam_WLSProxy         },*/
      /*{"WLSConfig",        GRB_StringParam_WLSConfig       },*/ {"Dummy",             GRB_StringParam_Dummy            }
   };
   const static array<string, 3> prefixes{"GRBD", "GRBI", "GRBS"};
   for(const auto& [name, args] : processed_opts) {
      if(name.starts_with("GRB")) {
         const string& param_name(name.substr(4));
         if     (name.starts_with("GRBD") && grb_double_params.contains(param_name)) {
            set_grb_double_param(env, args[0], grb_double_params.at(param_name));
         }
         else if(name.starts_with("GRBI") &&    grb_int_params.contains(param_name)) {
            set_grb_int_param(   env, args[0],    grb_int_params.at(param_name));
         }
         else if(name.starts_with("GRBS") && grb_string_params.contains(param_name)) {
            set_grb_string_param(env, args[0], grb_string_params.at(param_name));
         }
      }
   }
}

template<typename Func>
void process_yes_arg(
   const unordered_map< string, vec<string> >& opts, const string& arg_name, Func f
) {
   if(opts.contains(arg_name)) {
      f(opts.at(arg_name));
   }
}

template<typename Func>
void process_no_arg(
   const unordered_map< string, vec<string> >& opts, const string& arg_name, Func f
) {
   if(!opts.contains(arg_name)) {
      f();
   }
}

auto get_targets_paths(const path& load_path, const string& key_name) {
   vec<path> paths;
   ranges::move(
      directory_iterator(load_path) | views::filter([&key_name] (const auto& dir_entry) {
         return (
            dir_entry.is_regular_file()
            && (dir_entry.path().extension() <=> path(".csv")) == 0
            && dir_entry.path().filename().string().contains(key_name)
         );
      }),
      back_inserter(paths)
   );
   ranges::sort(paths);
   return paths;
}

auto get_targets(const RangeOf<path> auto& files_path, bool ignore_header = true, bool ignore_index = true) {
   vec<mat<double>> targets;
   ranges::move(files_path | views::transform([&ignore_header, &ignore_index] (const path& path) { 
      return read_matrix_from_csv<double>(fstream(path), ignore_header, ignore_index);
   }), back_inserter(targets));
   return targets;
}

template<typename T>
auto full_layer_parameter(const RangeOf<int> auto& cap, const T& value) {
   vec<mat<T>> data(ranges::distance(cap) - 1);
   for(auto&& [data, sizes] : views::zip(data, cap | views::adjacent<2>)) {
      const auto& [n, m] = sizes;
      data = mat<T>(n + 1);
      for(auto& row : data) {
         ranges::fill_n(back_inserter(row), m, value);
      }
   }
   return data;
}

template<typename G>
auto sample_data(
   const MatrixRange<double> auto& features,
   const Tensor3Range<double> auto& class_targets,
   const Tensor3Range<double> auto& regression_targets,
   int samples, bool shuffle, G& generator
) {
   auto total_instances = ranges::distance(features);
   for(const auto& tgts : class_targets) {
      total_instances = min(total_instances, ranges::distance(tgts));
   }
   for(const auto& tgts : regression_targets) {
      total_instances = min(total_instances, ranges::distance(tgts));
   }
   vec<int> indices(total_instances);
   ranges::iota(indices, 0);
   if(shuffle) {
      ranges::shuffle(indices, generator);
   }
   mat<double> new_features;
   ten3<double> new_cls_targets(ranges::distance(class_targets)), new_reg_targets(ranges::distance(regression_targets));
   ranges::copy(indices | views::take(samples) | views::transform([&features] (int i) {
      return features[i];
   }), back_inserter(new_features));
   for(auto&& [new_tgt, old_tgt] : views::zip(new_cls_targets, class_targets)) {
      new_tgt = mat<double>();
      ranges::copy(indices | views::take(samples) | views::transform([&old_tgt] (int i) {
         return old_tgt[i];
      }), back_inserter(new_tgt));
   }
   for(auto&& [new_tgt, old_tgt] : views::zip(new_reg_targets, regression_targets)) {
      new_tgt = mat<double>();
      ranges::copy(indices | views::take(samples) | views::transform([&old_tgt] (int i) {
         return old_tgt[i];
      }), back_inserter(new_tgt));
   }
   return make_tuple(new_features, new_cls_targets, new_reg_targets);
}

int main(int argc, const char* argv[]) try {
   // Procesa opciones y si no es correcta termina
   auto e_opts = process_opts(argc, argv);
   if(holds_alternative<string>(e_opts)) {
      cout << get<string>(e_opts);
      return 0;
   }
   auto opts = get<unordered_map<string,vec<string>>>(e_opts);
   // Parametros por default
   GRBEnv ambiente;
   // Procesa las rutas de los archivos a leer
   bool index       = !opts.contains("no_index" );
   bool header      = !opts.contains("no_header");
   path save_path   = opts.contains("save_path") ? opts["save_path"][0] : "";
   path load_path   = opts.contains("load_path") ? opts["load_path"][0] : "";
   string save_name = opts.contains("save_name") ? opts["save_name"][0] : "model";
   string load_name = opts.contains("load_name") ? opts["load_name"][0] : "";
   // Lee las caracteristicas de la arquitectura y la base de datos
   path arch_path     = load_path / format("{}.csv", safe_suffix(load_name, "arch"));
   path features_path = load_path / format("{}.csv", safe_suffix(load_name, "ftr"));
   const auto& regression_targets = get_targets(get_targets_paths(load_path, "reg_tgt"), header, index);
   const auto& class_targets      = get_targets(get_targets_paths(load_path, "cls_tgt"), header, index);
   const auto& [C, AF, bias, dropout, c_drop, l1w_norm, l1a_norm, l2w_norm, l2a_norm] = read_arch(fstream(arch_path));
   const auto& features = read_matrix_from_csv<double>(fstream(features_path), header, index);
   int L = C.size() - 1;
   cout << "Numero de capas " << L << "\n";
   for(const auto& [k, sizes] : C | views::adjacent<2> | views::enumerate) {
      const auto& [n, m] = sizes;
      cout << "Capa " << k << " matriz de pesos " << n + 1 << " x " << m << "\n"; 
   }
   // Lee los bits utilizados o crea uno por default
   int min_bits = 1;
   process_yes_arg(opts, "min_bits", [&min_bits](const auto& args) {
      stringstream(args[0]) >> min_bits;
   });
   int max_bits = 16;
   process_yes_arg(opts, "max_bits", [&max_bits](const auto& args) {
      stringstream(args[0]) >> max_bits;
   });
   int min_exp = -20;
   process_yes_arg(opts, "min_exp", [&min_exp](const auto& args) {
      stringstream(args[0]) >> min_exp;
   });
   int max_exp = 20;
   process_yes_arg(opts, "max_exp", [&max_exp](const auto& args) {
      stringstream(args[0]) >> max_exp;
   });
   vec<mat<int>> bits;
   if(opts.contains("bits")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "bits"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      bits = clamp_layers_matrix(get_layers_matrix<int, int>(data, C), min_bits, max_bits);
   }
   if(bits.empty()) {
      cout << "Trying with default bits\n";
      bits = full_layer_parameter<int>(C, max_bits);
   }
   // Lee la precision o exponente utilizado o crea uno por default
   vec<mat<int>> precision;
   if(opts.contains("exp")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "exp"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      precision = clamp_layers_matrix(get_layers_matrix<int, int>(data, C), min_exp, max_exp);
   }
   if(precision.empty()) {
      cout << "Trying with default exponent\n";
      precision = full_layer_parameter<int>(C, 2);
   }
   // Lee las mascaras utilizadas o crea una por default
   vec<mat<bool>> mask;
   if(opts.contains("mask")) {
      path file_path = load_path / format("{}.csv" ,safe_suffix(load_name, "mask"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      mask = get_layers_matrix<bool, int>(data, C);
   }
   if(mask.empty()) {
      cout << "Trying with default mask\n";
      mask = full_layer_parameter<bool>(C, true);
   }
   
   // Lee las mascaras utilizadas o crea una por default
   vec<mat<bool>> fixed;
   if(opts.contains("fixed")) {
      path file_path = load_path / format("{}.csv" ,safe_suffix(load_name, "fixed"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      fixed = get_layers_matrix<bool, int>(data, C);
   }
   if(fixed.empty()) {
      cout << "Trying with default fixed\n";
      fixed = full_layer_parameter<bool>(C, false);
   }
   // Lee los pesos iniciales o crea uno por default
   vec<mat<optional<double>>> init_w;
   if(opts.contains("init")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "init"));
      const auto& [dim, data] = read_list_from_csv<double>(fstream(file_path));
      init_w = get_layers_matrix<optional<double>, double>(data, C);
   }
   if(init_w.empty()) {
      cout << "Trying with default init\n";
      init_w = full_layer_parameter<optional<double>>(C, {});
   }
   // Lee los parametros de las funciones de activacion
   vec<vec<double>> params(L);
   if(opts.contains("params")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "params"));
      auto [dim, data] = read_list_from_csv<double>(fstream(file_path));
      ranges::move(data, params.begin());
   }
   // Genera variables para el ambiente
   string file_path = (save_path / safe_suffix(load_name, save_name)).string();
   string ResultFile = format("{}.sol.7z", file_path);
   string SolFiles = file_path;
   string LogFile = format("{}.log", file_path);
   // Procesa las opciones ingresadas
   auto rng = random_device();
   unsigned int seed = uniform_int_distribution(0, 2000000000)(rng);
   process_yes_arg(opts, "seed", [&seed, &ambiente](const auto& args) {
      stringstream(args[0]) >> seed;
   });
   double error_priority = 1.0;
   process_yes_arg(opts, "err_prio", [&error_priority](const auto& args) {
      stringstream(args[0]) >> error_priority;
   });
   optional<double> weight_penalty;
   process_yes_arg(opts, "wpen", [&weight_penalty](const auto& args) {
      stringstream(args[0]) >> weight_penalty;
   });
   double zero_tolerance = 0.0001;
   process_yes_arg(opts, "tol0", [&zero_tolerance](const auto& args) {
      stringstream(args[0]) >> zero_tolerance;
   });
   double constraint_tolerance = 0.1;
   process_yes_arg(opts, "reltol", [&constraint_tolerance](const auto& args) {
      stringstream(args[0]) >> constraint_tolerance;
   });
   double constraint_frac = 0.0;
   process_yes_arg(opts, "relax_frac", [&constraint_frac](const auto& args) {
      stringstream(args[0]) >> constraint_frac;
   });
   double zero_off_tolerance = 0.0;
   process_yes_arg(opts, "offtol", [&zero_off_tolerance](const auto& args) {
      stringstream(args[0]) >> zero_off_tolerance;
   });
   int lazy = 1;
   process_yes_arg(opts, "lazy", [&lazy](const auto& args) {
      stringstream(args[0]) >> lazy;
   });
   double offpen = 10;
   process_yes_arg(opts, "offpen", [&offpen](const auto& args) {
      stringstream(args[0]) >> offpen;
   });
   int samples = 10;
   process_yes_arg(opts, "samples", [&samples](const auto& args) {
      stringstream(args[0]) >> samples;
   });
   double bound = 10;
   process_yes_arg(opts, "bound", [&bound](const auto& args) {
      stringstream(args[0]) >> bound;
   });
   bool use_bound         =  opts.contains("bound"            );
   bool use_sos           =  opts.contains("sos"              );
   bool no_shuffle        =  opts.contains("no_shuffle"       );
   bool use_gurobi        =  opts.contains("grb_con"          );
   bool relaxed_model     =  opts.contains("relax_model"      );
   bool alternative_model =  opts.contains("alt_model"        );
   bool use_start         =  opts.contains("start"            );
   bool optimize          = !opts.contains("no_opti"          );
   bool save_lp           = !opts.contains("no_lp"            );
   bool save_ilp          = !opts.contains("no_ilp"           );
   bool save_sol          = !opts.contains("no_sol"           );
   bool save_mst          = !opts.contains("no_mst"           );
   bool save_json         = !opts.contains("no_json"          );
   bool no_dropouts       =  opts.contains("no_drop"          );
   bool no_l1             =  opts.contains("no_l1"            );
   bool no_l2             =  opts.contains("no_l2"            );
   bool use_square        =  opts.contains("square"           );
   bool hint              = !opts.contains("no_hint"          );
   int LogToConsole       = !opts.contains("no_log_to_console");
   process_no_arg(opts, "no_sols", [&SolFiles, &ambiente]() {
      ambiente.set(GRB_StringParam_SolFiles, SolFiles);
   });
   process_no_arg(opts, "no_log", [&LogFile, &ambiente]() {
      ambiente.set(GRB_StringParam_LogFile, LogFile);
   });
   ambiente.set(GRB_IntParam_JSONSolDetail, 1           );
   ambiente.set(GRB_IntParam_MIPFocus,      3           );
   ambiente.set(GRB_IntParam_Seed,          seed        );
   ambiente.set(GRB_IntParam_LogToConsole,  LogToConsole);
   mt19937 generator(seed);
   const auto& [ftrs, cls_tgts, reg_tgts] = sample_data(features, class_targets, regression_targets, samples, !no_shuffle, generator);
   int T = ranges::distance(ftrs);
   cout << "Numero de instancias " << T << "\n";
   // Genera los vectores de desactivar entradas o conexiones en la red neuronal, utilizando una semilla
   // o utilizado de manera aleatoria en caso de no proporcionarla
   const auto& [db, cdb] = generate_dropouts(C, dropout, c_drop, generator, no_dropouts);
   process_env(ambiente, opts);
   GRBModel modelo = get_model(
      /* GRBEnv ambiente gurobi */ambiente,
      /* Numero de casos T */ T,
      /* Numero de capas L */ L,
      /* Capacidad o numero de neuronas por capa */ C,
      /* Funciones de activacion */ AF,
      /* Mascaras de pesos */ mask,
      /* Precision para los pesos */ precision,
      /* Numero de bits para los pesos */bits,
      /* Valor inicial o esperado de los pesos */init_w,
      /* Valor de los umbrales o bias */ bias,
      /* Parametros de las funciones de activacion */ params,
      /* Probabilidades de volver cero alguna entrada */ db,
      /* Regularización L1 sobre activacion */ l1a_norm,
      /* Regularización L1 sobre pesos */ l1w_norm,
      /* Regularización L2 sobre activacion */ l2a_norm,
      /* Regularización L2 sobre pesos */ l2w_norm,
      /* Penalización por alejarse de la solución inicial */ weight_penalty,
      /* Mantener fijo los pesos */ fixed,
      /* Probabilidades de desactivar conexiones */ cdb,
      /* Matriz de las caracteristicas */ ftrs,
      /* Matriz de la regresion esperada */ reg_tgts,
      /* Matriz de la clasificacion deseada */ cls_tgts,
      /* Tolerancia utilizada en el logaritmo */ zero_tolerance,
      /* Porcentaje de casos usados para restricciones */ constraint_frac,
      /* Porcentaje de error o tolerancia sobre las restricciones */ constraint_tolerance,
      /* Tipo de restriccion en gurobi para restricciones no escenciales */ lazy,
      /* Usar penalizaciones en lugar de restricciones para mascaras y dropouts */ relaxed_model,
      /* Prioridad o importancia que se le da más al error que a otras regularizaciones */ error_priority,
      /* Usar inicio de los pesos iniciales */ use_start,
      /* Usar restricciones proporcionadas por gurobi */ use_gurobi,
      /* Usar restricciones tipo SOS1 */ use_sos,
      /* Usar tolerancia para los ceros en las matrices */ zero_off_tolerance,
      /* Usar modelo alternativo donde es un offset de los pesos iniciales */ alternative_model,
      /* Usar aproximacion de error cuadratico */ use_square,
      /* No usar L1 */ no_l1,
      /* No usar L2 */ no_l2,
      /* No usar las pistas */ hint,
      /* Utilizar la penalizacion por fuera del minimo o maximo */ offpen,
      use_bound,
      bound
   );
   if(save_lp)
      modelo.write(path(format("{}.lp.7z", file_path)).string());
   if(optimize) {
      modelo.update();
      modelo.optimize();
      switch(modelo.get(GRB_IntAttr_Status)) {
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
               modelo.computeIIS();
               modelo.write(format("{}.ilp.7z", file_path));
            }
            break;
         case GRB_UNBOUNDED :
            cout << "Modelo no acotado\n";
            break;
         default :
            cout << "Estado no manejado\n";
      }
   }
}
catch (const GRBException& ex) {
   cout << "No se pudo continuar\n";
   cout << ex.getMessage( ) << "\n";
}

// g++ grb_nn.cpp -std=c++23 -lgurobi_c++ -lgurobi110 -o grb_nn
// ./grb_nn --load_path [load_path] --save_path [save_path] --save_name [save_name] --load_name [load_name]
// <--use_bits> <--use_mask> <--use_exp> <--no_save_lp> <--no_optimize> <--no_save_mst> <--no_save_log>
// <--no_save_ilp> <--no_save_json> <--seed [number]> <--constr_tol [0.0-0.5]> <--relaxed_frac [0-1]>
// <--lazy [0-3]> <--zero_tol [number]> <--use_lrelu> <--use_init> <--no_log_to_console>
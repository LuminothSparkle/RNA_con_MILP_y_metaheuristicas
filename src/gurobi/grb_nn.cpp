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
T inverse_sigmoid(T x, T zero_tolerance = 0.000000000001) {
   return log(max(x, zero_tolerance)) - log1p(max(-x, zero_tolerance - 1));
}

string safe_suffix(const string & a, const string& b) {
   if(!a.ends_with("_") && b.compare("") != 0) {
      return a + "_" + b;
   }
   return a + b;
}

template<typename T>
istream& operator >>(istream& stream, optional<T>& opt) {
   T value;
   stream >> value;
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
      int exp;
      int sign = *w == 0 ? 0 : (1 - 2 * (*w < 0));
      double imantissa = frexp(*w, &exp) * exp2(size - 1);
      int inf = floor(imantissa * -sign) * sign;
      int sup = ceil(imantissa * -sign) * sign;
      for(auto&& [bf, bc] : views::zip(bits_floor, bits_ceil) | views::reverse) {
         bf = inf & 1;
         bc = sup & 1;
         inf >>= 1;
         sup >>= 1;
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
   int param;
   stringstream(arg) >> param;
   env.set(code, param);
}

void set_grb_double_param(GRBEnv& env, const string& arg, GRB_DoubleParam code) {
   double param;
   stringstream(arg) >> param;
   env.set(code, param);
}

void set_grb_string_param(GRBEnv& env, const string& arg, GRB_StringParam code) {
   string param;
   stringstream(arg) >> param;
   env.set(code, param);
}

/// @brief Analiza la expresion de gurobi para saber si solamente consiste
/// en una unica variable
/// @param expr Expresion a analizar
/// @return Verdadero si solo contiene una variable con coeficiente 1 y sin constante
bool is_single_expr(const GRBLinExpr& expr) {
   return expr.size() == 1 && expr.getConstant() == 0.0 && expr.getCoeff(0) == 1.0;
}

void set_hint(GRBVar& var, double value, int prio = -10) {
   var.set(GRB_DoubleAttr_VarHintVal, value);
   var.set(GRB_IntAttr_VarHintPri, prio);
}

void set_start(GRBModel& model, GRBVar& var, int start, double value) {
   model.set(GRB_IntParam_StartNumber, start);
   var.set(GRB_DoubleAttr_Start, value);
}

void gen_range_constr(
   GRBModel& model, const GRBLinExpr& expr, double min, double max,
   const string& constr_name, bool lazy = 1, bool use_gurobi = false
) {
   if(use_gurobi) {
      model.addRange(expr, min, max, constr_name).set(GRB_IntAttr_Lazy, lazy);
   }
   else {
      GRBVar var_u = model.addVar(0, max - min, 0, GRB_CONTINUOUS, format("{}_u", constr_name));
      model.addConstr(var_u + expr == max, constr_name).set(GRB_IntAttr_Lazy, lazy);
   }
}

auto gen_abs_obj_vars(GRBModel& model, const string& var_name, int var_hint = -10) {
   GRBVar plus = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_plus", var_name));
   GRBVar minus = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_minus", var_name));
   set_hint(plus, 0, var_hint);
   set_hint(minus, 0, var_hint);
   return make_tuple(plus, minus);
}

GRBLinExpr gen_abs_range_obj_expr(
   GRBModel& model, const GRBLinExpr& expr, const string& var_name, double min = 0, double max = 0,
   bool lazy = 1, bool use_gurobi = false, int var_hint = -10
) {
   const auto& [plus, minus] = gen_abs_obj_vars(model, var_name, var_hint);
   gen_range_constr(model, plus - minus + expr, min, max, var_name, lazy, use_gurobi);
   return plus + minus;
}

GRBLinExpr gen_approx_square_obj_expr(
   GRBModel& model, const GRBLinExpr& expr, const string& var_name,
   bool lazy = 1, bool use_gurobi = false, int var_hint = -10
) {
   auto [plus1, minus1] = gen_abs_obj_vars(model, format("{}_1", var_name), var_hint);
   plus1.set(GRB_DoubleAttr_UB, 2.0);
   minus1.set(GRB_DoubleAttr_UB, 2.0);
   auto [plus2, minus2] = gen_abs_obj_vars(model, format("{}_2", var_name), var_hint);
   plus2.set(GRB_DoubleAttr_UB, 4.5);
   minus2.set(GRB_DoubleAttr_UB, 4.5);
   auto [plus3, minus3] = gen_abs_obj_vars(model, format("{}_3", var_name), var_hint);
   gen_range_constr(model, plus1 - minus1 + plus2 - minus2 + plus3 - minus3 + expr, -0.5, 0.5, var_name, lazy, use_gurobi);
   return 2 * (plus1 + minus1) + 8 * (plus2 + minus2) + 20 * (plus3 + minus3);
}


GRBLinExpr gen_sum_expr(const GRBLinExprRange auto& X) {
   return accumulate(X.begin(), X.end(), GRBLinExpr());
}

GRBLinExpr gen_mean_expr(const GRBLinExprRange auto& X) {
   return gen_sum_expr(X) / max<int>(1, ranges::distance(X));
}

/// @brief Genera una variable de gurobi que es igual a una expresion. Añade la
/// restriccion y la varaible al modelo en caso de que no sea una expresion con
/// una unica variable, si es una expresion con una sola variable no genera nada
/// y devuelve la variable en cuestion para evitar generar más variables.
/// @param model El modelo de programacion en gurobi
/// @param expr Expresion para la que se genera la variable
/// @param var_name Nombre de la variable en caso de usarse
/// @param constr_name Nombre de la restriccion en caso de usarse
/// @param min Valor minimo de la variable en caso de usarse
/// @param max Valor maximo de la variable en caso de usarse
/// @param var_type Tipo de variable en caso de usarse
/// @param lazy Tipo de restriccion en caso de usarse
/// @return Regresa la variable que representa la expresion
GRBVar gen_var(
   GRBModel& model, const GRBLinExpr& expr, const string& var_name,
   double min = -GRB_INFINITY, double max = GRB_INFINITY,
   char var_type = GRB_CONTINUOUS, int lazy = 0
) {
   if(is_single_expr(expr)) {
      return expr.getVar(0);
   }
   GRBVar var = model.addVar(min, max, 0, var_type, var_name);
   model.addConstr(expr == var, var_name).set(GRB_IntAttr_Lazy, lazy);
   return var;
}

GRBVar gen_abs_var(
   GRBModel& model, const GRBLinExpr& x, const string& var_name,
   bool use_gurobi = false, bool objective = false, int lazy = 0
) {
   GRBVar var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   if(use_gurobi) {
      GRBVar new_x = gen_var(
         model, x,
         format("{}_in", var_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      model.addGenConstrAbs(var_abs, new_x, var_name);
   }
   else {
      model.addConstr(var_abs >= x, format("{}_plus", var_name)).set(GRB_IntAttr_Lazy, lazy);
      model.addConstr(var_abs >= -x, format("{}_minus", var_name)).set(GRB_IntAttr_Lazy, lazy);
      if(!objective) {
         GRBVar a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or", var_name));
         model.addGenConstrIndicator(a_or_b, 1, var_abs <= x, format("{}_on",  var_name));
         model.addGenConstrIndicator(a_or_b, 0, var_abs <= -x, format("{}_off", var_name));
      }
   }
   return var_abs;
}

GRBLinExpr gen_abs_expr(
   GRBModel& model, const GRBLinExpr& x, const string& var_name,
   bool use_gurobi = false, bool objective = false, int lazy = 0
) {
   return gen_abs_var(model, x, var_name, use_gurobi, objective, lazy);
}


GRBVar gen_bin_w_var(
   GRBModel& model, const GRBVar& b, const GRBLinExpr& a, const string& var_name,
   double coef, int lazy = 0, bool use_sos = false
) {
   GRBVar bw = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrIndicator(b, 0, bw == coef * a, format("{}_off", var_name));
   if(use_sos) {
      vec<GRBVar> vars_b({b, bw});
      vec<double> weights_b({0, 1});
      model.addSOS(vars_b.data(), weights_b.data(), 2, GRB_SOS_TYPE1);
   }
   else {
      model.addGenConstrIndicator(b, 1, bw == 0, format("{}_on",  var_name));
   }
   return bw;
}

GRBLinExpr gen_bin_w_expr(
   GRBModel& model, const GRBVar& b, const GRBLinExpr& a,
   const string& var_name, double coef, int lazy = 0, bool use_sos = false
) {
   return gen_bin_w_var(model, b, a, var_name, coef, lazy, use_sos);
}

GRBLinExpr gen_hardtanh_expr(
   GRBModel& model, const GRBLinExpr& z, const string& constr_name,
   const pair<double, double>& limits = {-1,1}, int lazy = 0,
   bool use_gurobi = false, bool use_sos = false
) {
   GRBVar zmax_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_max", constr_name));
   GRBVar zmin_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_min", constr_name));
   if(!use_gurobi) {
      model.addConstr(zmax_max >= z - limits.second, format("{}_maxub", constr_name)).set(GRB_IntAttr_Lazy, lazy);
      model.addConstr(zmin_max >= z - limits.first, format("{}_minub", constr_name)).set(GRB_IntAttr_Lazy, lazy);
      GRBVar on_zmax = model.addVar(0, 1, 0, GRB_BINARY, format("{}_maxlb", constr_name));
      model.addGenConstrIndicator(on_zmax, 0, zmax_max <= z - limits.second, format("{}_maxoff",  constr_name));
      GRBVar on_zmin = model.addVar(0, 1, 0, GRB_BINARY, format("{}_minlb", constr_name));
      model.addGenConstrIndicator(on_zmin, 0, zmin_max <= z - limits.first, format("{}_minoff",  constr_name));
      if(use_sos) {
         array<GRBVar, 2> vars_zmax{on_zmax, zmax_max}; array<double, 2> weights_zmax{0, 1};
         model.addSOS(vars_zmax.data(), weights_zmax.data(), 2, GRB_SOS_TYPE1);
         array<GRBVar, 2> vars_zmin{on_zmin, zmin_max}; array<double, 2> weights_zmin{0, 1};
         model.addSOS(vars_zmin.data(), weights_zmin.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(on_zmax, 1, zmax_max <= 0, format("{}_maxon",  constr_name));
         model.addGenConstrIndicator(on_zmin, 1, zmin_max <= 0, format("{}_minon",  constr_name));
      }
   }
   else {
      GRBVar zmax = gen_var(
         model, z - limits.second,
         format("{}_maxin", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      GRBVar zmin = gen_var(
         model, z - limits.first,
         format("{}_minin", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      model.addGenConstrMax(zmin_max, &zmax, 1, 0, format("{}_max", constr_name));
      model.addGenConstrMax(zmax_max, &zmin, 1, 0, format("{}_min", constr_name));
   }
   return limits.first + zmin_max - zmax_max;
}

GRBLinExpr gen_hardsigmoid_expr(
   GRBModel& model, const GRBLinExpr& z, const string& constr_name,
   int lazy = 0, bool use_gurobi = false, bool use_sos = false
) {
   GRBLinExpr new_z = z / 6 + 0.5;
   GRBVar z_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_z", constr_name));
   GRBVar z1_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_z1", constr_name));
   if(!use_gurobi) {
      model.addConstr(z_max >= new_z, format("{}_zub", constr_name)).set(GRB_IntAttr_Lazy, lazy);
      GRBVar on_z = model.addVar(0, 1, 0, GRB_BINARY, format("{}_zlb", constr_name));
      model.addGenConstrIndicator(on_z, 0, z_max <= new_z, format("{}_zoff",  constr_name));
      model.addConstr(z1_max >= new_z - 1, format("{}_z1ub", constr_name)).set(GRB_IntAttr_Lazy, lazy);
      GRBVar on_z1 = model.addVar(0, 1, 0, GRB_BINARY, format("{}_z1lb", constr_name));
      model.addGenConstrIndicator(on_z1, 0, z1_max <= new_z - 1, format("{}_z1off",  constr_name));
      if(use_sos) {
         array<GRBVar, 2> vars_z{on_z, z_max};
         array<double, 2> weights_z{0, 1};
         model.addSOS(vars_z.data(), weights_z.data(), 2, GRB_SOS_TYPE1);
         array<GRBVar, 2> vars_z1{on_z1, z1_max};
         array<double, 2> weights_z1{0, 1};
         model.addSOS(vars_z1.data(), weights_z1.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(on_z, 1, z_max <= 0, format("{}_zon",  constr_name));
         model.addGenConstrIndicator(on_z1, 1, z1_max <= 0, format("{}_z1on",  constr_name));
      }
      
   }
   else {
      GRBVar new_z0 = gen_var(
         model, new_z,
         format("{}_zin", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      GRBVar new_z1 = gen_var(
         model, new_z - 1,
         format("{}_z1in", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      model.addGenConstrMax(z_max, &new_z0, 1, 0, format("{}_z", constr_name));
      model.addGenConstrMax(z1_max, &new_z1, 1, 0, format("{}_z1", constr_name));
   }
   return z_max - z1_max;
}

GRBLinExpr gen_ReLU6_expr(
   GRBModel& model, const GRBLinExpr& z, const string& constr_name,
   int lazy = 0, bool use_gurobi = false, bool use_sos = false
) {
   GRBVar z_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_z", constr_name));
   GRBVar z6_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_z6", constr_name));
   if(!use_gurobi) {
      model.addConstr(z_max >= z, format("{}_zub", constr_name)).set(GRB_IntAttr_Lazy, lazy);
      GRBVar on_z = model.addVar(0, 1, 0, GRB_BINARY, format("{}_zlb", constr_name));
      model.addGenConstrIndicator(on_z, 0, z_max <= z, format("{}_zoff",  constr_name));
      model.addConstr(z6_max >= z - 6, format("{}_z6ub", constr_name)).set(GRB_IntAttr_Lazy, lazy);
      GRBVar on_z6 = model.addVar(0, 1, 0, GRB_BINARY, format("{}_z6lb", constr_name));
      model.addGenConstrIndicator(on_z6, 0, z6_max <= z - 6, format("{}_z6off",  constr_name));
      if(use_sos) {
         array<GRBVar, 2> vars_z{on_z, z_max}; array<double, 2> weights_z{0, 1};
         model.addSOS(vars_z.data(), weights_z.data(), 2, GRB_SOS_TYPE1);
         array<GRBVar, 2> vars_z6{on_z6, z6_max}; array<double, 2> weights_z6{0, 1};
         model.addSOS(vars_z6.data(), weights_z6.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(on_z, 1, z_max <= 0, format("{}_zon",  constr_name));
         model.addGenConstrIndicator(on_z6, 1, z6_max <= 0, format("{}_z6on",  constr_name));
      }
   }
   else {
      GRBVar new_z = gen_var(
         model, z,
         format("{}_zin", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      GRBVar new_z6 = gen_var(
         model, z - 6,
         format("{}_z6in", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      model.addGenConstrMax(z_max, &new_z, 1, 0, format("{}_z", constr_name));
      model.addGenConstrMax(z6_max, &new_z6, 1, 0, format("{}_z6", constr_name));
   }
   return z_max - z6_max;
}

GRBLinExpr gen_ReLU_expr(
   GRBModel& model, const GRBLinExpr& z, const string& constr_name,
   int lazy = 0, bool use_gurobi = false, bool use_sos = false
) {
   GRBVar var = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, constr_name);
   if(!use_gurobi) {
      model.addConstr(var >= z, format("{}_ub", constr_name)).set(GRB_IntAttr_Lazy, lazy);
      GRBVar on_z = model.addVar(0, 1, 0, GRB_BINARY, format("{}_lb", constr_name));
      model.addGenConstrIndicator(on_z, 0, var <= z, format("{}_off",  constr_name));
      if(use_sos) {
         array<GRBVar, 2> vars_z{on_z, var}; array<double, 2> weights_z{0, 1};
         model.addSOS(vars_z.data(), weights_z.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(on_z, 1, var <= 0, format("{}_on",  constr_name));
      }
   }
   else {
      GRBVar new_z = gen_var(
         model, z,
         format("{}_in", constr_name),
         -GRB_INFINITY, GRB_INFINITY,
         GRB_CONTINUOUS, lazy
      );
      model.addGenConstrMax(var, &new_z, 1, 0, constr_name);
   }
   return var;
}

GRBLinExpr gen_LeakyReLU_expr(
   GRBModel& model, const GRBLinExpr& z, const string& constr_name,
   double neg_coef = 0.25, int lazy = 0, bool use_gurobi = false
) {
   GRBLinExpr z_abs = gen_abs_expr(model, z, constr_name, use_gurobi, false, lazy);
   GRBLinExpr min_z0 = (z - z_abs) / 2;
   GRBLinExpr max_z0 = (z + z_abs) / 2;
   return max_z0 + neg_coef * min_z0;
}

GRBLinExpr gen_act_w_expr(const GRBLinExprRange auto& bw) {
   return accumulate(bw.begin() + 1, bw.end(), -bw[0]);
}

GRBLinExpr gen_w_expr(
   GRBModel& model, const GRBLinExprRange auto& b, int exponent = 4,
   int lazy = 0, GRBLinExpr tw = 0
) {
   vec<GRBLinExpr> exp;
   for(const auto& [l, b] : b | views::enumerate) {
      exp.emplace_back(exp2(exponent - l) * (1 - b));
   }
   return accumulate(exp.begin() + 1, exp.end(), -exp[0] + tw);
}

GRBVar gen_w_var(
   GRBModel& model, const GRBLinExprRange auto& b, const string& var_name,
   int precis = 4, int lazy = 0, GRBLinExpr tw = 0
) {
   return gen_var(
      model,
      gen_w_expr(model, b, precis, lazy, tw),
      var_name,
      -GRB_INFINITY, GRB_INFINITY,
      GRB_CONTINUOUS, lazy
   );
}

auto gen_activation_exprs(
   GRBModel& model, const string& type, const GRBLinExprRange auto& z,
   const string& constr_name, const RangeOf<double> auto& LeakyReLU_coef,
   const pair<double,double>& hardtanh_limits,
   int lazy = 0, bool use_gurobi = false, bool use_sos = false
) {
   vec<GRBLinExpr> a;
   if(type == "ReLU") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_ReLU_expr(
            model, z,
            format("{}_relu_{}", constr_name, j),
            lazy, use_gurobi, use_sos
         ));
      }
   }
   else if(type == "ReLU6") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_ReLU6_expr(
            model, z,
            format("{}_relu6_{}", constr_name, j),
            lazy, use_gurobi, use_sos
         ));
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_LeakyReLU_expr(
            model, z,
            format("{}_lrelu_{}", constr_name, j),
            LeakyReLU_coef[j], lazy, use_gurobi
         ));
      }
   }
   else if(type  == "Hardtanh") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_hardtanh_expr(
            model, z,
            format("{}_htanh_{}", constr_name, j),
            hardtanh_limits, lazy, use_gurobi, use_sos
         ));
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& [j, z] : z | views::enumerate) {
         a.emplace_back(gen_hardsigmoid_expr(
            model, z,
            format("{}_hsig_{}", constr_name, j),
            lazy, use_gurobi, use_sos
         ));
      }
   }
   else {
      ranges::move(z, back_inserter(a));
   } 
   return a;
}

template<typename R>
requires RangeOf<R, double> && ranges::random_access_range<R>
GRBLinExpr gen_class_error_expr(
   GRBModel& model, const GRBLinExprRange auto& y, const string& constr_name, const RangeOf<double> auto& ty,
   int cases, const R& class_count, double zero_tolerance = 0.000000000001,
   double constraint_tolerance = 0.1, int lazy = 0, bool use_gurobi = false
) {
   vec<GRBLinExpr> exprs;
   if(ty.size() == 1) {
      // Clasificación binaria
      double ty0 = *ranges::begin(ty);
      GRBLinExpr y0 = *ranges::begin(y);
      double divisor = ty0 < 0.5 ? cases - class_count[0] : class_count[0];
      double min_value = inverse_sigmoid(max(-constraint_tolerance + ty0, ty0 > 0.5 ? 0.5 : 0), zero_tolerance);
      double max_value = inverse_sigmoid(min(constraint_tolerance + ty0, ty0 < 0.5 ? 0.5 : 1), zero_tolerance);
      exprs.emplace_back(
         gen_abs_range_obj_expr(
            model, y0, constr_name, min_value, max_value, lazy, use_gurobi, 10
         ) / max(zero_tolerance, divisor)
      );
   }
   else if(ty.size() > 1) {
      // Clasificación multiclase
      double min_prob = 1.0 / ty.size();
      GRBVar c = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_C",constr_name));
      for(const auto& [ty, y, cls] : views::zip( ty, y, views::iota(0))) {
         double divisor = class_count[cls];
         double min_value = log(max(max(-constraint_tolerance + ty, ty > min_prob ? min_prob : 0), zero_tolerance));
         double max_value = log(max(min(constraint_tolerance + ty, ty < min_prob ? min_prob : 1), zero_tolerance)); 
         exprs.emplace_back(gen_abs_range_obj_expr(
            model, y - c, format("{}_{}", constr_name, cls), min_value, max_value, lazy, use_gurobi, 10
         ) / max(zero_tolerance, divisor));
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_regression_error_expr(
   GRBModel& model, const GRBLinExprRange auto& y, const string& constr_name,
   const RangeOf<double> auto& ty, double zero_tolerance = 0.0001,
   double constraint_tolerance = 0.0, int lazy = 0, bool use_gurobi = false, bool use_square = false
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [i, y, ty] : views::zip(views::iota(0), y, ty)) {
      if(!use_square) {
         double min_value = -constraint_tolerance * max(abs(ty), zero_tolerance);
         double max_value = constraint_tolerance * max(abs(ty), zero_tolerance);
         exprs.emplace_back(gen_abs_range_obj_expr(
            model, y - ty, format("{}_{}", constr_name, i),
            min_value, max_value, lazy, use_gurobi, 10
         ));
      }
      else {
         exprs.emplace_back(gen_approx_square_obj_expr(
            model, y - ty, format("{}_{}", constr_name, i),
            lazy, use_gurobi, 10
         ));
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l1w_expr(
   GRBModel& model, const MatrixRange<GRBVar> auto& w, const string& constr_name,
   int lazy = 0, double constr_tol = 0, bool use_gurobi = false
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [i, w] : w | views::enumerate) {
      for(const auto& [j, w] : w | views::enumerate) {
         exprs.emplace_back(gen_abs_range_obj_expr(
            model, w, format("{}_{}_{}", constr_name, i, j),
            -constr_tol, constr_tol, lazy, use_gurobi, -10
         ));
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l2w_expr(
   GRBModel& model, const MatrixRange<GRBVar> auto& w,
   const string& constr_name, int lazy = 0, bool use_gurobi = false
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [i, w] : w | views::enumerate) {
      for(const auto& [j, w] : w | views::enumerate) {
         exprs.emplace_back(gen_approx_square_obj_expr(
            model, w, format("{}_{}_{}", constr_name, i, j),
            lazy, use_gurobi, -10
         ));
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l1a_expr(
   GRBModel& model, const GRBLinExprRange auto& a, const string& constr_name,
   int lazy = 0, double constr_tol = 0, bool use_gurobi = false
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [j, a] : a | views::enumerate) {
      exprs.emplace_back(gen_abs_range_obj_expr(
         model, a, format("{}_{}", constr_name, j),
         -constr_tol, constr_tol, lazy, use_gurobi, -10
      ));
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l2a_expr(
   GRBModel& model, const GRBLinExprRange auto& a,
   const string& constr_name, int lazy = 0, bool use_gurobi = false
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [j, a] : a | views::enumerate) {
      exprs.emplace_back(gen_approx_square_obj_expr(
         model, a, format("{}_{}", constr_name, j),
         lazy, use_gurobi, -10
      ));
   }
   return gen_mean_expr(exprs);
}


GRBLinExpr gen_wp_expr(
   GRBModel& model, const MatrixRange<GRBVar> auto& w, const MatrixRange<optional<double>> auto& tw,
   const string& constr_name, int lazy = 0, double constr_tol = 0,
   double zero_tolerance = 0.00001, bool use_gurobi = false, bool use_square = false
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [i, w, tw] : views::zip(views::iota(0), w, tw)) {
      for(const auto& [j, w, tw] : views::zip(views::iota(0), w, tw)) {
         if(tw) {
            if(!use_square) {
               double min_value = -constr_tol * max(abs(*tw), zero_tolerance);
               double max_value = constr_tol * max(abs(*tw), zero_tolerance);
               exprs.emplace_back(gen_abs_range_obj_expr(
                  model, w - *tw, format("{}_{}_{}", constr_name, i, j),
                  min_value, max_value, lazy, use_gurobi, 5
               ));
            }
            else {
               exprs.emplace_back(gen_approx_square_obj_expr(
                  model, w - *tw, format("{}_{}_{}", constr_name, i, j),
                  lazy, use_gurobi, 5
               ));
            }
         }
      }
   }
   return gen_mean_expr(exprs);
}

auto gen_class_count(const Tensor3Range<double> auto& tc_y) {
   vec<vec<double>> class_count;
   for(auto& tc_y : tc_y) {
      class_count.emplace_back(vec<double>(ranges::distance(*ranges::begin(tc_y)), 0.0));
      for(const auto& tc_y : tc_y) {
         for(auto&& [tc_y, class_count] : views::zip(tc_y, class_count.back())) {
            class_count += tc_y;
         }
      }
   }
   return class_count;
}

auto gen_layers(
   GRBModel& model, int L, const RangeOf<int> auto& C, const Tensor3Range<bool> auto& mask,
   const Tensor3Range<int> auto& precis, const Tensor3Range<int> auto& D,
   const Tensor3Range<optional<double>> auto& tw, const RangeOf<optional<double>> auto& l1w_norm,
   const RangeOf<optional<double>> auto& l2w_norm, const optional<double>& w_pen,
   const Tensor3Range<bool> auto& fixed, int lazy = 1, bool relaxed_model = false,
   bool use_start = false, bool use_gurobi = false, double zero_off_tolerance = 0.0,
   double zero_tolerance = 0.000001, double relative_tolerance = 0.0,
   bool alternative_model = false, bool use_square = false, bool no_l1 = false, bool no_l2 = false
) {
   GRBVar one_var = model.addVar(1, 1, 0, GRB_BINARY, "one_singleton");
   GRBVar zero_var = model.addVar(0, 0, 0, GRB_BINARY, "zero_singleton");
   set_hint(one_var, 1, 20);
   set_hint(zero_var, 0, 20);
   if(use_start) {
      int starts = 2;
      if(relaxed_model && alternative_model) {
         starts += 1;
      }
      else if(!relaxed_model && !alternative_model) {
         starts += 1;
      }
      else if(relaxed_model && !alternative_model) {
         starts += 2;
      }
      resetline_console();
      cout << "Starts = " << starts << "\n";
      model.set(GRB_IntAttr_NumStart, starts);
      for(int i = 0; i < starts; ++i) {
         set_start(model, zero_var, i, 0);
         set_start(model, one_var, i, 1);
      }
   }
   ten4<GRBVar> b(L); ten3<GRBVar> iw(L);
   vec<GRBLinExpr> mask_exprs, l1w_exprs, l2w_exprs, w_pen_exprs;
   for(auto&& [k, b, tw, bits, mask, exp, fixed, l1w_norm, l2w_norm, iw] : views::zip(views::iota(0), b, tw, D, mask, precis, fixed, l1w_norm, l2w_norm, iw)) {
      b = ten3<GRBVar>(C[k] + 1, mat<GRBVar>(C[k + 1]));
      iw = mat<GRBVar>(C[k] + 1, vec<GRBVar>(C[k + 1]));
      mat<GRBVar>    w(C[k] + 1, vec<GRBVar>(C[k + 1]));
      for(   auto&& [i, b, w, tw, bits, exp, mask, fixed, iw] : views::zip(views::iota(0), b, w, tw, bits, exp, mask, fixed, iw)) {
         for(auto&& [j, b, w, tw, bits, exp, mask, fixed, iw] : views::zip(views::iota(0), b, w, tw, bits, exp, mask, fixed, iw)) {
            if(!tw) {
               model.set(GRB_IntAttr_NumStart, 1);
            }
            const auto& [bin_floor, bin_ceil] = decompose_w(tw, bits + 1);
            b = vec<GRBVar>(bits + 1);
            if(fixed) {
               ranges::fill(b, one_var);
               iw = zero_var;
            }
            else if(!relaxed_model && !mask) {
               ranges::fill(b, one_var);
               iw = one_var;
            }
            else {
               for(auto&& [l, b, bf, bc] : views::zip(views::iota(0), b, bin_floor, bin_ceil)) {
                  b = model.addVar(
                     0, 1, 0, GRB_BINARY,
                     format("b_{}_{}_{}_{}", k, i, j, l)
                  );
                  if(use_start) {
                     set_start(model, b, 0, 1);
                     if(tw && relaxed_model && alternative_model) {
                        set_start(model, b, 1, 1);
                        set_start(model, b, 2, 1);
                     }
                     else if(tw && !relaxed_model && alternative_model) {
                        set_start(model, b, 1, 1);
                     }
                     else if(tw && !relaxed_model && !alternative_model) {
                        set_start(model, b, 1, (1 - *bf) * mask);
                        set_start(model, b, 2, (1 - *bc) * mask);
                     }
                     else if(tw && relaxed_model && !alternative_model) {
                        set_start(model, b, 1, (1 - *bf));
                        set_start(model, b, 2, (1 - *bc));
                        set_start(model, b, 3, (1 - *bf) * mask);
                        set_start(model, b, 4, (1 - *bc) * mask);
                     }
                  }
               }
               if(tw && alternative_model) {
                  iw = model.addVar(0, 1, 0, GRB_BINARY, format("iw_{}_{}_{}", k, i, j));
                  if(use_start) {
                     set_start(model, iw, 0, 1);
                     if(relaxed_model) {
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
               model, b,
               format("w_{}_{}_{}", k, i, j),
               exp, 0, tw.value_or(0.0) * (1 - iw)
            );
            if(relaxed_model && !mask) {
               if(!use_square) {
                  mask_exprs.emplace_back(gen_abs_range_obj_expr(
                     model, w, format("mask_{}_{}_{}", k, i, j),
                     -zero_off_tolerance, zero_off_tolerance, lazy, use_gurobi, 5
                  ));
               }
               else {
                  mask_exprs.emplace_back(gen_approx_square_obj_expr(
                     model, w, format("mask_{}_{}_{}", k, i, j),
                     lazy, use_gurobi, 5
                  ));
               }
            }
            else if(!relaxed_model && !mask) {
               w.set(GRB_DoubleAttr_LB, 0);
               w.set(GRB_DoubleAttr_UB, 0);
               set_hint(w, 0, 10);
            }
            if(use_start) {
               set_start(model, w, 0, 0);
               if(tw && relaxed_model && alternative_model) {
                  set_start(model, w, 1, *tw);
                  set_start(model, w, 2, *tw * mask);
               }
               else if(tw && !relaxed_model && alternative_model) {
                  set_start(model, w, 1, *tw * mask);
               }
               else if(tw && !relaxed_model && !alternative_model) {
                  set_start(model, w, 1, calculate_w(bin_floor, exp) * mask);
                  set_start(model, w, 2, calculate_w(bin_ceil, exp) * mask);
               }
               else if(tw && relaxed_model && !alternative_model) {
                  set_start(model, w, 1, calculate_w(bin_floor, exp));
                  set_start(model, w, 2, calculate_w(bin_ceil, exp));
                  set_start(model, w, 3, calculate_w(bin_floor, exp) * mask);
                  set_start(model, w, 4, calculate_w(bin_ceil, exp) * mask);
               }
            } 
         }
      }
      if(w_pen) {
         w_pen_exprs.emplace_back(*w_pen * gen_wp_expr(
            model, w, tw,
            format("wp_{}", k),
            lazy, relative_tolerance, zero_tolerance,
            use_gurobi, use_square
         ));
      }
      if(!no_l1 && l1w_norm && k + 1 < L) {
         l1w_exprs.emplace_back(*l1w_norm * gen_l1w_expr(
            model, w, 
            format("l1w_{}", k),
            lazy, zero_off_tolerance, use_gurobi
         ));
      }
      
      if(!no_l2 && l2w_norm && k + 1 < L) {
         l2w_exprs.emplace_back(*l2w_norm * gen_l2w_expr(
            model, w, 
            format("l2w_{}", k),
            lazy, use_gurobi
         ));
      }
   }
   return make_tuple(b, mask_exprs, l1w_exprs, l2w_exprs, w_pen_exprs, iw);
}

auto gen_dropout_exprs(
   GRBModel& model, const GRBLinExprRange auto& a, const RangeOf<bool> auto& dropout,
   const string& constr_name, bool relaxed_model = false, int lazy = 1,
   double dropout_tolerance = 0.0, bool use_gurobi = true
) {
   GRBLinExpr obj_expr; vec<GRBLinExpr> a_exprs;
   if(relaxed_model) {
      vec<GRBLinExpr> pen_dropout;
      for(const auto& [i, a, dropout] : views::zip(views::iota(0), a, dropout)) {
         if(dropout) {
            pen_dropout.emplace_back(gen_abs_range_obj_expr(
               model, a, format("{}_{}", constr_name, i),
               -dropout_tolerance, dropout_tolerance, lazy, use_gurobi, 5
            ));
         }
         a_exprs.emplace_back(a);
      }
      obj_expr = gen_mean_expr(pen_dropout);
   }
   else {
      for(const auto& [a, dropout] : views::zip(a , dropout)) {
         a_exprs.emplace_back(dropout ? GRBLinExpr() : GRBLinExpr(a));
      }
   }
   return make_tuple(a_exprs, obj_expr);
}

auto gen_connection_dropout_exprs(
   GRBModel& model, const GRBLinExprRange auto& a, const Tensor3Range<GRBVar> auto& b,
   int n, int m, const MatrixRange<bool> auto& c_drop, const MatrixRange<bool> auto& mask,
   const MatrixRange<int> auto& exp, const MatrixRange<optional<double>> auto& tw,
   const MatrixRange<GRBVar> auto& iw, const string& constr_name, bool relaxed_model = false,
   int lazy = 1, double dropout_tolerance = 0.0, bool use_gurobi = true,
   bool alternative_model = false, bool use_square = false
) {
   GRBLinExpr obj_expr; mat<GRBLinExpr> aw_exprs(m, vec<GRBLinExpr>(n + 1));
   if(relaxed_model) {
      vec<GRBLinExpr> pen_dropout_exprs;
      for(const auto& [i, b, a, exp, c_drop, mask, tw, iw] : views::zip(views::iota(0), b, a, exp, c_drop, mask, tw, iw)) {
         for(const auto& [j, b, exp, c_drop, mask, tw, iw] : views::zip(views::iota(0), b,    exp, c_drop, mask, tw, iw)) {
            if(tw && alternative_model) {
               aw_exprs[j][i] += gen_bin_w_var(
                  model, iw, a,
                  format("{}_{}_{}_w", constr_name, i, j),
                  *tw, lazy
               );
            }
            vec<GRBLinExpr> bin_w;
            for(const auto& [l, b] : b | views::enumerate) {
               bin_w.emplace_back(gen_bin_w_var(
                  model, b, a,
                  format("{}_{}_{}_{}", constr_name, i, j, l),
                  exp2(exp - l), lazy
               ));
            }
            aw_exprs[j][i] += gen_act_w_expr(bin_w);
            if(c_drop) {
               if(!use_square) {
                  pen_dropout_exprs.emplace_back(gen_abs_range_obj_expr(
                     model, aw_exprs[j][i], format("{}_{}_{}_condrop", constr_name, i, j),
                     -dropout_tolerance, dropout_tolerance, lazy, use_gurobi, 5
                  ));
               }
               else {
                  pen_dropout_exprs.emplace_back(gen_approx_square_obj_expr(
                     model, aw_exprs[j][i],
                     format("{}_{}_{}_condrop", constr_name, i, j),
                     lazy, use_gurobi, 5
                  ));
               }
            }
         }
      }
   }
   else {
      for(const auto& [i, b, a, exp, c_drop, mask, tw, iw] : views::zip(views::iota(0), b, a, exp, c_drop, mask, tw, iw)) {
         for(const auto& [j, b, exp, c_drop, mask, tw, iw] : views::zip(views::iota(0), b,    exp, c_drop, mask, tw, iw)) {
            if(mask && !c_drop) {
               if(tw && alternative_model) {
                  aw_exprs[j][i] += gen_bin_w_var(
                     model, iw, a,
                     format("{}_{}_{}_w", constr_name , i, j),
                     *tw, lazy
                  );
               }
               vec<GRBLinExpr> bin_w;
               for(const auto& [l, b] : b | views::enumerate) {
                  bin_w.emplace_back(gen_bin_w_var(
                     model, b, a,
                     format("{}_{}_{}_{}", constr_name, i, j, l),
                     exp2(exp - l), lazy
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
   const GRBEnv& environment, int T, int L, const RangeOf<int> auto& C, const RangeOf<string> auto& AF, 
   const Tensor3Range<bool> auto& mask, const Tensor3Range<int> auto& precis, const Tensor3Range<int> auto& D,
   const Tensor3Range<optional<double>> auto& tw, const RangeOf<double> auto& bias_w,
   const MatrixRange<double> auto& leakyReLU_coef, const RangeOf<pair<double,double>> auto& hardtanh_limits,
   const MatrixRange<bool> auto& dropout, const RangeOf<optional<double>> auto& l1a_norm,
   const RangeOf<optional<double>> auto& l1w_norm, const RangeOf<optional<double>> auto& l2a_norm,
   const RangeOf<optional<double>> auto& l2w_norm, const optional<double>& w_pen,
   const Tensor3Range<bool> auto& fixed, const Tensor3Range<bool> auto& c_drop,
   const MatrixRange<double> auto& fx, const Tensor3Range<double> auto& reg_ty,
   const Tensor3Range<double> auto& class_ty, double zero_tolerance = 0.000000000001,
   double relaxed_frac = 0.0, double relative_tolerance = 0.1, int lazy = 1, bool relaxed_model = false,
   double error_priority = 1.0, bool use_start = false, bool use_gurobi = false, bool use_sos = false,
   double zero_off_tolerance = 0.0, bool alternative_model = false, bool use_square = false,
   bool no_l1 = false, bool no_l2 = false
) {
   GRBModel model(environment);
   resetline_console();
   cout << "Processing layers variables..." << flush;
   const auto& [b, mask_exprs, l1w_exprs, l2w_exprs, w_pen_exprs, iw] = gen_layers(
      model, L, C, mask, precis, D, tw, l1w_norm, l2w_norm, w_pen, fixed, lazy, relaxed_model, use_start,
      use_gurobi, zero_off_tolerance, zero_tolerance, relative_tolerance, alternative_model, use_square,
      no_l1, no_l2
   );
   vec<GRBLinExpr> dropout_exprs, c_drop_exprs, l1a_exprs, l2a_exprs, target_exprs;
   auto class_count = gen_class_count(class_ty);
   int relaxed_size = relaxed_frac * T;
   for(const auto& [t, fx] : fx | views::enumerate) {
      resetline_console();
      cout << "Processing case: " << t << "...\n" << flush;
      vec<GRBLinExpr> a;
      ranges::copy(fx, back_inserter(a));
      vec<GRBLinExpr> case_l1a_exprs, case_l2a_exprs, case_target_exprs;
      for(const auto& [k, b, AF, exp, dropout, leakyReLU_coef, hardtanh_limits, c_drop, l1a_norm, l2a_norm, bias_w, mask, tw, iw] : views::zip(
         views::iota(0), b, AF, precis, dropout, leakyReLU_coef, hardtanh_limits, c_drop, l1a_norm, l2a_norm, bias_w, mask, tw, iw
      )) {
         resetline_console();
         cout << "Processing layer: " << k << "..." << flush;
         a.emplace_back(bias_w);
         const auto& [drop_a, drop_expr] = gen_dropout_exprs(
            model, a, dropout, format("drop_{}_{}", t, k),
            relaxed_model, lazy, zero_off_tolerance, use_gurobi
         );
         dropout_exprs.emplace_back(drop_expr);
         const auto& [z, c_drop_expr] = gen_connection_dropout_exprs(
            model, drop_a, b, C[k], C[k + 1], c_drop, mask, exp, tw, iw,
            format("bw_{}_{}", t, k), relaxed_model, lazy,
            zero_off_tolerance, use_gurobi, alternative_model
         );
         c_drop_exprs.emplace_back(c_drop_expr);
         a = gen_activation_exprs(
            model, AF, z, format("act_{}_{}", t, k), leakyReLU_coef,
            hardtanh_limits, lazy, use_gurobi, use_sos
         );
         if(!no_l1 &&l1a_norm && k + 1 < L) {
            case_l1a_exprs.emplace_back(*l1a_norm * gen_l1a_expr(
               model, a, format("l1a_{}_{}", t, k),
               lazy, zero_off_tolerance, use_gurobi
            ));
         }
         if(!no_l2 && l2a_norm && k + 1 < L) {
            case_l2a_exprs.emplace_back(*l2a_norm * gen_l2a_expr(
               model, a, format("l2a_{}_{}", t, k),
               lazy, use_gurobi
            ));
         }
      }
      int size_accumulated = 0;
      for(const auto& [ti, class_ty, class_count] : views::zip(views::iota(0), class_ty, class_count)) {
         if(class_ty[t].size() > 0) {
            vec<GRBLinExpr> cy;
            ranges::move(a | views::drop(size_accumulated) | views::take(class_ty[t].size()), back_inserter(cy));
            case_target_exprs.emplace_back(gen_class_error_expr(
               model, cy,
               format("cls_{}_{}", t, ti),
               class_ty[t], T, class_count, zero_tolerance,
               t < relaxed_size ? relative_tolerance : 0.0, lazy, use_gurobi
            ));
            size_accumulated += class_ty[t].size();
         }
      }
      for(const auto& [ti, reg_ty] : reg_ty | views::enumerate) {
         if (reg_ty[t].size() > 0) {
            vec<GRBLinExpr> ty;
            ranges::move(
               a | views::drop(size_accumulated) |
               views::take(reg_ty[t].size()), back_inserter(ty)
            );
            case_target_exprs.emplace_back(gen_regression_error_expr(
               model, ty,
               format("reg_{}_{}", t, ti),
               reg_ty[t], zero_tolerance, t < relaxed_size ? relative_tolerance : 0.0,
               lazy, use_gurobi, use_square
            ));
            size_accumulated += reg_ty[t].size();
         }
      }
      if(case_l2a_exprs.size() > 0) {
         l2a_exprs.emplace_back(gen_mean_expr(case_l2a_exprs));
      }
      if(case_l1a_exprs.size() > 0) {
         l1a_exprs.emplace_back(gen_mean_expr(case_l1a_exprs));
      }
      if(case_target_exprs.size() > 0) {
         target_exprs.emplace_back(gen_mean_expr(case_target_exprs));
      }
      cursorup_console(1);
   }
   cursorup_console(1);
   model.setObjective(
      T * L * (
         error_priority * gen_mean_expr(target_exprs) +
         gen_mean_expr(l1a_exprs) + gen_mean_expr(l1w_exprs) +
         gen_mean_expr(l2a_exprs) + gen_mean_expr(l2w_exprs) +
         gen_mean_expr(w_pen_exprs) + gen_mean_expr(dropout_exprs) +
         gen_mean_expr(c_drop_exprs) + gen_mean_expr(mask_exprs)
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
   const RangeOf<int> auto& capacity, const RangeOf<optional<double>> auto& dropout,
   const RangeOf<optional<double>> auto& connection_dropout, G&& generator, bool no_dropouts = false
) {
   vec<vec<bool>> dropout_bool(dropout.size());
   vec<mat<bool>> c_dropout_bool(connection_dropout.size());
   for(auto&& [k, db, pd, cdb, cpd] : views::zip(
      views::iota(0), dropout_bool, dropout, c_dropout_bool, connection_dropout
   )) {
      db = vec<bool>(capacity[k] + 1, false);
      cdb = mat<bool>(capacity[k] + 1);
      if(!no_dropouts) {
         ranges::generate(db, gen_dropout_func(generator, pd));
      }
      for(auto& cdb : cdb) {
         cdb = vec<bool>(capacity[k + 1], false);
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
   vec<int> C; vec<string> AF; vec<double> bias; vec<pair<double,double>> HT;
   if(ignore_header) {
      getline(input, line);
   }
   while(getline(input, line)) {
      stringstream line_stream(line);
      if(ignore_index) {
         getline(line_stream, word, ',');
      }
      int k; optional<string> af;
      optional<double> l1a, l1w, l2a, l2w, drop, b, ht_min, ht_max, cd;
      getline(line_stream, word, ','); stringstream(word) >> k;
      getline(line_stream, word, ','); stringstream(word) >> af;
      getline(line_stream, word, ','); stringstream(word) >> drop;
      getline(line_stream, word, ','); stringstream(word) >> ht_min;
      getline(line_stream, word, ','); stringstream(word) >> ht_max;
      getline(line_stream, word, ','); stringstream(word) >> l1w;
      getline(line_stream, word, ','); stringstream(word) >> l1a;
      getline(line_stream, word, ','); stringstream(word) >> l2w;
      getline(line_stream, word, ','); stringstream(word) >> l2a;
      getline(line_stream, word, ','); stringstream(word) >> b;
      getline(line_stream, word, ','); stringstream(word) >> cd;
      C.emplace_back(k);
      AF.emplace_back(af.value_or("None"));
      Drop.emplace_back(drop);
      HT.emplace_back(make_pair(ht_min.value_or(-1), ht_max.value_or(1)));
      L1w.emplace_back(l1w);
      L1a.emplace_back(l1a);
      L2w.emplace_back(l2w);
      L2a.emplace_back(l2a);
      bias.emplace_back(b.value_or(1.0));
      c_drop.emplace_back(cd);
   }
   return make_tuple(C, AF, Drop, HT, L1w, L1a, L2w, L2a, bias, c_drop);
}

auto read_arch(istream&& input, bool ignore_index = false, bool ignore_header = true) {
   return read_arch(input, ignore_index, ignore_header);
}

template<typename R, typename T>
auto get_layers_matrix(const RangeOf<vec<int>> auto& dim, const MatrixRange<T> auto& data) {
   vec<mat<R>> layers_data;
   for(const auto& [dim, data] : views::zip(dim, data)) {
      int n = dim[0], m = dim[1];
      mat<R> layer_data(n);
      for(int i = 0; i < n; ++i) {
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
      {"load_path",         1},
      {"load_name",         1},
      {"save_path",         1},
      {"save_name",         1},
      {"no_log_to_console", 0},
      {"no_save_sols",      0},
      {"no_save_log",       0},
      {"no_save_json",      0},
      {"no_save_sol",       0},
      {"no_save_mst",       0},
      {"no_save_ilp",       0},
      {"no_save_lp",        0},
      {"no_optimize",       0},
      {"use_exp",           0},
      {"use_mask",          0},
      {"use_lrelu",         0},
      {"use_fixed",         0},
      {"use_bits",          0},
      {"no_l1",             0},
      {"no_l2",             0},
      {"no_dropouts",       0},
      {"zero_tol",          1},
      {"constr_tol",        1},
      {"relaxed_frac",      1},
      {"error_prio",        1},
      {"relaxed_model",     0},
      {"use_init",          0},
      {"use_start",         0},
      {"gurobi_constrs",    0},
      {"alternative_model", 0},
      {"use_sos",           0},
      {"use_square",        0},
      {"min_bits",          1},
      {"max_bits",          1},
      {"min_exp",           1},
      {"max_exp",           1},
      {"lazy",              1},
      {"zero_off_tol",      1},
      {"weight_penalty",    1},
      {"seed",              1}
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
      {"SolutionLimit",    GRB_IntParam_SolutionLimit   }, {"Method",             GRB_IntParam_Method            },
      {"ConcurrentMethod", GRB_IntParam_ConcurrentMethod}, {"ScaleFlag",          GRB_IntParam_ScaleFlag         },
      {"SimplexPricing",   GRB_IntParam_SimplexPricing  }, {"Quad",               GRB_IntParam_Quad              },
      {"NormAdjust",       GRB_IntParam_NormAdjust      }, {"Sifting",            GRB_IntParam_Sifting           },
      {"SiftMethod",       GRB_IntParam_SiftMethod      }, {"NetworkAlg",         GRB_IntParam_NetworkAlg        },
      {"LPWarmStart",      GRB_IntParam_LPWarmStart     }, {"SubMIPNodes",        GRB_IntParam_SubMIPNodes       },
      {"VarBranch",        GRB_IntParam_VarBranch       }, {"Cuts",               GRB_IntParam_Cuts              },
      {"CliqueCuts",       GRB_IntParam_CliqueCuts      }, {"CoverCuts",          GRB_IntParam_CoverCuts         },
      {"FlowCoverCuts",    GRB_IntParam_FlowCoverCuts   }, {"FlowPathCuts",       GRB_IntParam_FlowPathCuts      },
      {"GUBCoverCuts",     GRB_IntParam_GUBCoverCuts    }, {"ImpliedCuts",        GRB_IntParam_ImpliedCuts       },
      /*{"DualImpliedCuts",  GRB_IntParam_DualImpliedCuts },*/ {"ProjImpliedCuts",    GRB_IntParam_ProjImpliedCuts   },
      {"MIPSepCuts",       GRB_IntParam_MIPSepCuts      }, {"MIRCuts",            GRB_IntParam_MIRCuts           },
      {"StrongCGCuts",     GRB_IntParam_StrongCGCuts    }, {"ModKCuts",           GRB_IntParam_ModKCuts          },
      {"ZeroHalfCuts",     GRB_IntParam_ZeroHalfCuts    }, {"NetworkCuts",        GRB_IntParam_NetworkCuts       },
      {"SubMIPCuts",       GRB_IntParam_SubMIPCuts      }, {"InfProofCuts",       GRB_IntParam_InfProofCuts      },
      {"RelaxLiftCuts",    GRB_IntParam_RelaxLiftCuts   }, {"RLTCuts",            GRB_IntParam_RLTCuts           },
      {"BQPCuts",          GRB_IntParam_BQPCuts         }, {"PSDCuts",            GRB_IntParam_PSDCuts           },
      {"LiftProjectCuts",  GRB_IntParam_LiftProjectCuts }, {"MixingCuts",         GRB_IntParam_MixingCuts        },
      {"CutAggPasses",     GRB_IntParam_CutAggPasses    }, {"CutPasses",          GRB_IntParam_CutPasses         },
      {"GomoryPasses",     GRB_IntParam_GomoryPasses    }, {"NodeMethod",         GRB_IntParam_NodeMethod        },
      {"Presolve",         GRB_IntParam_Presolve        }, {"Aggregate",          GRB_IntParam_Aggregate         },
      {"IISMethod",        GRB_IntParam_IISMethod       }, {"PreCrush",           GRB_IntParam_PreCrush          },
      {"PreDepRow",        GRB_IntParam_PreDepRow       }, {"PrePasses",          GRB_IntParam_PrePasses         },
      {"DisplayInterval",  GRB_IntParam_DisplayInterval }, {"OutputFlag",         GRB_IntParam_OutputFlag        },
      {"Threads",          GRB_IntParam_Threads         }, {"BarIterLimit",       GRB_IntParam_BarIterLimit      },
      {"Crossover",        GRB_IntParam_Crossover       }, {"CrossoverBasis",     GRB_IntParam_CrossoverBasis    },
      {"BarCorrectors",    GRB_IntParam_BarCorrectors   }, {"BarOrder",           GRB_IntParam_BarOrder          },
      {"PumpPasses",       GRB_IntParam_PumpPasses      }, {"RINS",               GRB_IntParam_RINS              },
      {"Symmetry",         GRB_IntParam_Symmetry        }, {"MIPFocus",           GRB_IntParam_MIPFocus          },
      {"NumericFocus",     GRB_IntParam_NumericFocus    }, {"AggFill",            GRB_IntParam_AggFill           },
      {"PreDual",          GRB_IntParam_PreDual         }, {"SolutionNumber",     GRB_IntParam_SolutionNumber    },
      {"MinRelNodes",      GRB_IntParam_MinRelNodes     }, {"ZeroObjNodes",       GRB_IntParam_ZeroObjNodes      },
      {"BranchDir",        GRB_IntParam_BranchDir       }, {"DegenMoves",         GRB_IntParam_DegenMoves        },
      {"InfUnbdInfo",      GRB_IntParam_InfUnbdInfo     }, {"DualReductions",     GRB_IntParam_DualReductions    },
      {"BarHomogeneous",   GRB_IntParam_BarHomogeneous  }, {"PreQLinearize",      GRB_IntParam_PreQLinearize     },
      {"MIQCPMethod",      GRB_IntParam_MIQCPMethod     }, {"NonConvex",          GRB_IntParam_NonConvex         },
      {"QCPDual",          GRB_IntParam_QCPDual         }, {"LogToConsole",       GRB_IntParam_LogToConsole      },
      {"PreSOS1Encoding",  GRB_IntParam_PreSOS1Encoding }, {"PreSOS2Encoding",    GRB_IntParam_PreSOS2Encoding   },
      {"PreSparsify",      GRB_IntParam_PreSparsify     }, {"PreMIQCPForm",       GRB_IntParam_PreMIQCPForm      },
      {"Seed",             GRB_IntParam_Seed            }, {"ConcurrentMIP",      GRB_IntParam_ConcurrentMIP     },
      {"ConcurrentJobs",   GRB_IntParam_ConcurrentJobs  }, {"DistributedMIPJobs", GRB_IntParam_DistributedMIPJobs},
      {"LazyConstraints",  GRB_IntParam_LazyConstraints }, {"TuneResults",        GRB_IntParam_TuneResults       },
      {"TuneTrials",       GRB_IntParam_TuneTrials      }, {"TuneOutput",         GRB_IntParam_TuneOutput        },
      {"TuneJobs",         GRB_IntParam_TuneJobs        }, {"TuneCriterion",      GRB_IntParam_TuneCriterion     },
      {"TuneMetric",       GRB_IntParam_TuneMetric      }, {"TuneDynamicJobs",    GRB_IntParam_TuneDynamicJobs   },
      {"Disconnected",     GRB_IntParam_Disconnected    }, {"UpdateMode",         GRB_IntParam_UpdateMode        },
      {"Record",           GRB_IntParam_Record          }, {"ObjNumber",          GRB_IntParam_ObjNumber         },
      {"MultiObjMethod",   GRB_IntParam_MultiObjMethod  }, {"MultiObjPre",        GRB_IntParam_MultiObjPre       },
      {"PoolSolutions",    GRB_IntParam_PoolSolutions   }, {"PoolSearchMode",     GRB_IntParam_PoolSearchMode    },
      {"ScenarioNumber",   GRB_IntParam_ScenarioNumber  }, {"StartNumber",        GRB_IntParam_StartNumber       },
      {"StartNodeLimit",   GRB_IntParam_StartNodeLimit  }, {"IgnoreNames",        GRB_IntParam_IgnoreNames       },
      {"PartitionPlace",   GRB_IntParam_PartitionPlace  }, {"CSPriority",         GRB_IntParam_CSPriority        },
      {"CSTLSInsecure",    GRB_IntParam_CSTLSInsecure   }, {"CSIdleTimeout",      GRB_IntParam_CSIdleTimeout     },
      {"ServerTimeout",    GRB_IntParam_ServerTimeout   }, {"TSPort",             GRB_IntParam_TSPort            },
      {"JSONSolDetail",    GRB_IntParam_JSONSolDetail   }, {"CSBatchMode",        GRB_IntParam_CSBatchMode       },
      {"FuncPieces",       GRB_IntParam_FuncPieces      }, {"CSClientLog",        GRB_IntParam_CSClientLog       },
      {"IntegralityFocus", GRB_IntParam_IntegralityFocus}, {"NLPHeur",            GRB_IntParam_NLPHeur           },
      {"WLSTokenDuration", GRB_IntParam_WLSTokenDuration}, {"LicenseID",          GRB_IntParam_LicenseID         },
      {"OBBT",             GRB_IntParam_OBBT            }, {"FuncNonlinear",      GRB_IntParam_FuncNonlinear     },
      {"SolutionTarget",   GRB_IntParam_SolutionTarget  }/*, {"ThreadLimit",        GRB_IntParam_ThreadLimit       }*/
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
      {"LogFile",        GRB_StringParam_LogFile       }, {"NodefileDir",    GRB_StringParam_NodefileDir   },
      {"ResultFile",     GRB_StringParam_ResultFile    }, {"WorkerPool",     GRB_StringParam_WorkerPool    },
      {"WorkerPassword", GRB_StringParam_WorkerPassword}, {"ComputeServer",  GRB_StringParam_ComputeServer },
      {"ServerPassword", GRB_StringParam_ServerPassword}, {"CSRouter",       GRB_StringParam_CSRouter      },
      {"CSGroup",        GRB_StringParam_CSGroup       }, {"TokenServer",    GRB_StringParam_TokenServer   },
      {"CloudAccessID",  GRB_StringParam_CloudAccessID }, {"CloudSecretKey", GRB_StringParam_CloudSecretKey},
      {"CloudPool",      GRB_StringParam_CloudPool     }, {"CloudHost",      GRB_StringParam_CloudHost     },
      {"JobID",          GRB_StringParam_JobID         }, {"CSManager",      GRB_StringParam_CSManager     },
      {"CSAuthToken",    GRB_StringParam_CSAuthToken   }, {"CSAPIAccessID",  GRB_StringParam_CSAPIAccessID },
      {"CSAPISecret",    GRB_StringParam_CSAPISecret   }, {"UserName",       GRB_StringParam_UserName      },
      {"CSAppName",      GRB_StringParam_CSAppName     }, {"SolFiles",       GRB_StringParam_SolFiles      },
      {"WLSAccessID",    GRB_StringParam_WLSAccessID   }, {"WLSSecret",      GRB_StringParam_WLSSecret     },
      {"WLSToken",       GRB_StringParam_WLSToken      }, /*{"WLSProxy",       GRB_StringParam_WLSProxy      },*/
      /*{"WLSConfig",      GRB_StringParam_WLSConfig     },*/ {"Dummy",          GRB_StringParam_Dummy         }
   };
   const static array<string, 3> prefixes{"GRBD", "GRBI", "GRBS"};
   for(const auto& [name, args] : processed_opts) {
      const string& param_name(name.substr(4));
      if(name.starts_with("GRBD") && grb_double_params.contains(param_name)) {
         set_grb_double_param(env, args[0], grb_double_params.at(param_name));
      }
      else if(name.starts_with("GRBI") && grb_int_params.contains(param_name)) {
         set_grb_int_param(env, args[0], grb_int_params.at(param_name));
      }
      else if(name.starts_with("GRBS") && grb_string_params.contains(param_name)) {
         set_grb_string_param(env, args[0], grb_string_params.at(param_name));
      }
   }
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

auto get_targets(const RangeOf<path> auto& files_path) {
   vec<mat<double>> targets;
   ranges::move(files_path | views::transform([](const path& path) { 
      return read_matrix_from_csv<double>(fstream(path));
   }), back_inserter(targets));
   return targets;
}

template<typename T>
auto full_layer_parameter(const RangeOf<int> auto& C, const T& value) {
   vec<mat<T>> data(C.size() - 1);
   for(const auto& [k, sizes] : C | views::adjacent<2> | views::enumerate) {
      const auto& [n, m] = sizes;
      data[k] = mat<T>(n + 1);
      for(auto& row : data[k]) {
         ranges::fill_n(back_inserter(row), m, value);
      }
   }
   return data;
}

int main(int argc, const char* argv[]) try {
   // Procesa opciones y si no es correcta termina
   auto e_opts = process_opts(argc,argv);
   if(holds_alternative<string>(e_opts)) {
      cout << get<string>(e_opts);
      return 0;
   }
   auto opts = get<unordered_map<string,vec<string>>>(e_opts);
   // Parametros por default
   GRBEnv ambiente;
   // Procesa las rutas de los archivos a leer
   path save_path   = opts.contains("save_path") ? opts["save_path"][0] : "";
   path load_path   = opts.contains("load_path") ? opts["load_path"][0] : "";
   string save_name = opts.contains("save_name") ? opts["save_name"][0] : "model";
   string load_name = opts.contains("load_name") ? opts["load_name"][0] : "";
   // Lee las caracteristicas de la arquitectura y la base de datos
   path arch_path     = load_path / format("{}.csv", safe_suffix(load_name, "arch"));
   path features_path = load_path / format("{}.csv", safe_suffix(load_name, "ftr"));
   const auto& regression_targets = get_targets(get_targets_paths(load_path, "reg_tgt"));
   const auto& class_targets      = get_targets(get_targets_paths(load_path, "cls_tgt"));
   const auto& [C, AF, dropout, hardtanh, l1w_norm, l1a_norm, l2w_norm, l2a_norm, bias, c_drop] = read_arch(fstream(arch_path));
   const auto& features = read_matrix_from_csv<double>(fstream(features_path), true, true);
   int T = features.size(), L = C.size() - 1;
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
   if(opts.contains("use_bits")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "bits"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() > 0 && dim[0].size() == 2) {
         bits = clamp_layers_matrix(get_layers_matrix<int, int>(dim, data), min_bits, max_bits);
      }
   }
   if(bits.empty()) {
      bits = full_layer_parameter<int>(C, 4);
   }
   // Lee la precision o exponente utilizado o crea uno por default
   vec<mat<int>> precision;
   if(opts.contains("use_exp")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "exp"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() > 0 && dim[0].size() == 2) {
         precision = clamp_layers_matrix(get_layers_matrix<int, int>(dim, data), min_exp, max_exp);
      }
   }
   if(precision.empty()) {
      precision = full_layer_parameter<int>(C, 2);
   }
   // Lee las mascaras utilizadas o crea una por default
   vec<mat<bool>> mask;
   if(opts.contains("use_mask")) {
      path file_path = load_path / format("{}.csv" ,safe_suffix(load_name, "mask"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() > 0 && dim[0].size() == 2) {
         mask = get_layers_matrix<bool, int>(dim, data);
      }
   }
   if(mask.empty()) {
      mask = full_layer_parameter<bool>(C, true);
   }
   
   // Lee las mascaras utilizadas o crea una por default
   vec<mat<bool>> fixed;
   if(opts.contains("use_fixed")) {
      path file_path = load_path / format("{}.csv" ,safe_suffix(load_name, "fixed"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() > 0 && dim[0].size() == 2) {
         fixed = get_layers_matrix<bool, int>(dim, data);
      }
   }
   if(fixed.empty()) {
      fixed = full_layer_parameter<bool>(C, false);
   }
   // Lee los pesos iniciales o crea uno por default
   vec<mat<optional<double>>> init_w;
   if(opts.contains("use_init")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "init"));
      const auto& [dim, data] = read_list_from_csv<double>(fstream(file_path));
      if(dim.size() > 0 && dim[0].size() == 2) {
         init_w = get_layers_matrix<optional<double>, double>(dim, data);
      }
   }
   if(init_w.empty()) {
      init_w = full_layer_parameter<optional<double>>(C, {});
   }
   // Lee el leakyrelu o crea uno con valores por default
   vec<vec<double>> leakyReLU;
   if(opts.contains("use_lrelu")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "lrelu"));
      const auto& [dim, data] = read_list_from_csv<double>(fstream(file_path));
      leakyReLU = data;
   }
   if(leakyReLU.empty()) {
      leakyReLU = vec<vec<double>>(L);
      for(auto&& [leakyReLU, C] : views::zip(leakyReLU, C | views::drop(1))) {
         ranges::fill_n(back_inserter(leakyReLU), C, 0.25);
      }
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
   process_yes_arg(opts, "error_prio", [&error_priority](const auto& args) {
      stringstream(args[0]) >> error_priority;
   });
   optional<double> weight_penalty;
   process_yes_arg(opts, "weight_penalty", [&weight_penalty](const auto& args) {
      stringstream(args[0]) >> weight_penalty;
   });
   double zero_tolerance = 0.0001;
   process_yes_arg(opts, "zero_tol", [&zero_tolerance](const auto& args) {
      stringstream(args[0]) >> zero_tolerance;
   });
   double constraint_tolerance = 0.1;
   process_yes_arg(opts, "constr_tol", [&constraint_tolerance](const auto& args) {
      stringstream(args[0]) >> constraint_tolerance;
   });
   double constraint_frac = 0.0;
   process_yes_arg(opts, "relaxed_frac", [&constraint_frac](const auto& args) {
      stringstream(args[0]) >> constraint_frac;
   });
   double zero_off_tolerance = 0.0;
   process_yes_arg(opts, "zero_off_tol", [&zero_off_tolerance](const auto& args) {
      stringstream(args[0]) >> zero_off_tolerance;
   });
   int lazy = 1;
   process_yes_arg(opts, "lazy", [&lazy](const auto& args) {
      stringstream(args[0]) >> lazy;
   });
   bool use_sos           =  opts.contains("use_sos"          );
   bool use_gurobi        =  opts.contains("gurobi_constrs"   );
   bool relaxed_model     =  opts.contains("relaxed_model"    );
   bool alternative_model =  opts.contains("alternative_model");
   bool use_start         =  opts.contains("use_start"        );
   bool optimize          = !opts.contains("no_optimize"      );
   bool save_lp           = !opts.contains("no_save_lp"       );
   bool save_ilp          = !opts.contains("no_save_ilp"      );
   bool save_sol          = !opts.contains("no_save_sol"      );
   bool save_mst          = !opts.contains("no_save_mst"      );
   bool save_json         = !opts.contains("no_save_json"     );
   bool no_dropouts       =  opts.contains("no_dropouts"      );
   bool no_l1             =  opts.contains("no_l1"            );
   bool no_l2             =  opts.contains("no_l2"            );
   bool use_square        =  opts.contains("use_square"       );
   int LogToConsole       = !opts.contains("no_log_to_console");
   process_no_arg(opts, "no_save_sols", [&SolFiles, &ambiente]() {
      ambiente.set(GRB_StringParam_SolFiles, SolFiles);
   });
   process_no_arg(opts, "no_save_log", [&LogFile, &ambiente]() {
      ambiente.set(GRB_StringParam_LogFile, LogFile);
   });
   ambiente.set(GRB_IntParam_JSONSolDetail, 1           );
   ambiente.set(GRB_IntParam_MIPFocus,      3           );
   ambiente.set(GRB_IntParam_Seed,          seed        );
   ambiente.set(GRB_IntParam_LogToConsole,  LogToConsole);
   // Genera los vectores de desactivar entradas o conexiones en la red neuronal, utilizando una semilla
   // o utilizado de manera aleatoria en caso de no proporcionarla
   const auto& [db, cdb] = generate_dropouts(C, dropout, c_drop, mt19937(seed), no_dropouts);
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
      /* Coeficientes de LeakyReLU*/ leakyReLU,
      /* Valores maximos y minimos de hardtanh */ hardtanh,
      /* Probabilidades de volver cero alguna entrada */ db,
      /* Regularización L1 sobre activacion */ l1a_norm,
      /* Regularización L1 sobre pesos */ l1w_norm,
      /* Regularización L2 sobre activacion */ l2a_norm,
      /* Regularización L2 sobre pesos */ l2w_norm,
      /* Penalización por alejarse de la solución inicial */ weight_penalty,
      /* Mantener fijo los pesos */ fixed,
      /* Probabilidades de desactivar conexiones */ cdb,
      /* Matriz de las caracteristicas */ features,
      /* Matriz de la clasificacion deseada */ class_targets,
      /* Matriz de la regresion esperada */ regression_targets,
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
      /* No usar L2 */ no_l2
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
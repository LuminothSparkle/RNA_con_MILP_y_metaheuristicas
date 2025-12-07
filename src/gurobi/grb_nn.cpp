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
#include <numbers>

// Simplificacion de espacio de nombres
namespace fsys = std::filesystem;   namespace ranges = std::ranges;
namespace views = std::views;
// Manejo de rutas
using fsys::path;                    using fsys::directory_iterator;
// Funciones y tipos de cadenas
using std::format;                   using std::string;
using std::stoi;                     using std::stod;
// Algoritmos de conveniencia
using std::accumulate;               using std::clamp;
// Funciones de utilidad
using std::reference_wrapper;        using std::move;
using std::forward;
// Contenedores o utilidades
using std::make_tuple;               using std::make_pair;
using std::unordered_map;            using std::optional;
using std::pair;                     using std::tuple;
using std::variant;                  using std::holds_alternative;
using std::monostate;                using std::array;
// Streams y relacionados utilizados
using std::flush;                    using std::cin;
using std::clog;                     using std::cout;
using std::stringstream;             using std::istringstream;
using std::fstream;                  using std::istream;
using std::ostream;                  using std::getline;
using std::ios_base;
// Funciones de random utilizadas
using std::uniform_int_distribution; using std::random_device;
using std::bernoulli_distribution;   using std::mt19937;
// Funciones de cmath utilizadas
using std::tan;                      using std::numbers::pi_v;
using std::pow;                      using std::atanh;
using std::log10;                    using std::fma;
using std::log;                      using std::log1p;
using std::exp2;                     using std::exp;
using std::abs;                      using std::frexp;
using std::ldexp;                    using std::modf;
using std::max;                      using std::min;
using std::signbit;
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

double logit(double x, double eps) {
   return log(max(x, eps)) - log1p(max(-x, eps - 1.0));
}

double sigmoid(double x) {
   return 0.5 * (1 + tanh(x / 2.0));
}

double cross_entropy_loss(double x, double y, double w = 1.0, double eps = 2.09e-9) {
   return -w * y * log(max<double>(x, eps));
}

double binary_cross_entropy_loss(double x, double y, double w = 1.0, double recall = 1.0, double eps = 2.09e-9) {
   return -w * (y * recall * log(max<double>(x, eps)) + (1 - y) * log(max<double>(1 - x, eps)));
}

double mse_loss(double x, double y) {
   return (x - y) * (x - y);
}

double l1_loss(double x, double y) {
   return abs(x - y);
}

double smooth_l1_loss(double x, double y, double beta = 1.0) {
   double a = abs(x - y);
   return a < beta ? 0.5 * a * a / beta : a - 0.5 * beta;
}

double huber_loss(double x, double y, double delta = 1.0) {
   double a = abs(x - y);
   return a < delta ? 0.5 * a * a : delta * (a - 0.5 * delta);
}

double kldivergence_loss(double x, double y, double eps = 2.09e-9) {
   return y * (log(max<double>(y, eps)) - log(max<double>(x, eps)));
}

double limit_bits(double x, int bits) {
   int exp, a;
   double new_mantissa = frexp(round(ldexp(frexp(x, &exp), bits)), &a);
   return ldexp(new_mantissa, exp);
}

void resetline_console() {
   cout << "\x0D\x1b[K" << flush;
}

void cursorup_console(int n) {
   cout << "\x1b[" << n << "F" << flush;
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

void set_priority(GRBVar& var, int priority) {
   var.set(GRB_IntAttr_BranchPriority, priority);
   var.set(GRB_IntAttr_Partition,      priority);
}

void set_constr(GRBModel& model, const GRBTempConstr& constr, const string& name, int lazy) {
   model.addConstr(constr, name).set(GRB_IntAttr_Lazy, lazy);
}

double get_ub(const GRBVar& x) {
   return x.get(GRB_DoubleAttr_UB);
}

double get_lb(const GRBVar& x) {
   return x.get(GRB_DoubleAttr_LB);
}

void set_ub(GRBVar& x, double value) {
   x.set(GRB_DoubleAttr_UB, value);
}

void set_lb(GRBVar& x, double value) {
   x.set(GRB_DoubleAttr_LB, value);
}

void set_bounds(GRBVar& x, double lb, double ub) {
   set_lb(x, lb);   set_ub(x, ub);
}

auto get_bounds(const GRBVar& x) {
   return make_pair(get_lb(x), get_ub(x));
}

auto minus_bounds(const pair<double, double>& bounds) {
   return make_pair(-bounds.second, -bounds.first);
}

auto add_bounds(const pair<double, double>& bounds_a, const pair<double, double>& bounds_b) {
   return make_pair(bounds_a.first + bounds_b.first, bounds_a.second + bounds_b.second);
}

auto sub_bounds(const pair<double, double>& bounds_a, const pair<double, double>& bounds_b) {
   return make_pair(bounds_a.first - bounds_b.second, bounds_a.second - bounds_b.first);
}

auto add_bounds(const pair<double, double>& bounds_a, double b) {
   return make_pair(bounds_a.first + b, bounds_a.second + b);
}

auto max_bounds(const pair<double, double>& bounds_a, const pair<double, double>& bounds_b) {
   return make_pair(max<double>(bounds_a.first, bounds_b.first), max<double>(bounds_a.second, bounds_b.second));
}

auto min_bounds(const pair<double, double>& bounds_a, const pair<double, double>& bounds_b) {
   return make_pair(min<double>(bounds_a.first, bounds_b.first), min<double>(bounds_a.second, bounds_b.second));
}

auto max_bounds(const pair<double, double>& bounds_a, double b) {
   return make_pair(max<double>(bounds_a.first, b), max<double>(bounds_a.second, b));
}

auto min_bounds(const pair<double, double>& bounds_a, double b) {
   return make_pair(min<double>(bounds_a.first, b), min<double>(bounds_a.second, b));
}

auto or_bounds(const pair<double, double>& bounds_a, double b) {
   return make_pair(min<double>(b, bounds_a.first), max<double>(b, bounds_a.second));
}

auto or_bounds(const pair<double, double>& bounds_a, const pair<double, double>& bounds_b) {
   return make_pair(min<double>(bounds_b.first, bounds_a.first), max<double>(bounds_b.second, bounds_a.second));
}

auto and_bounds(const pair<double, double>& bounds_a, const pair<double, double>& bounds_b) {
   return make_pair(max<double>(bounds_b.first, bounds_a.first), min<double>(bounds_b.second, bounds_a.second));
}

auto abs_bounds(const pair<double, double>& bounds) {
   double lb_abs = abs(bounds.first), ub_abs = abs(bounds.second);
   if(signbit(bounds.first) == signbit(bounds.second)) {
      return make_pair(min<double>(lb_abs, ub_abs), max<double>(lb_abs, ub_abs));
   }
   return make_pair(0.0, max<double>(lb_abs, ub_abs));
}

auto add_zero_bounds(const pair<double, double>& bounds) {
   return or_bounds(bounds, 0.0);
}

auto mult_bounds(const pair<double, double>& bounds_a, double b) {
   if(b > 0) {
      return make_pair(b * bounds_a.first, b * bounds_a.second);
   }
   else if(b < 0)  {
      return make_pair(b * bounds_a.second, b * bounds_a.first);
   }
   return make_pair(0.0, 0.0);
}

auto mult_bounds(const pair<double, double>& bounds_a, const pair<double, double>& bounds_b) {
   const auto& a_b_1 = mult_bounds(bounds_a, bounds_b.first);
   const auto& a_b_2 = mult_bounds(bounds_a, bounds_b.second);
   return make_pair(min<double>(a_b_1.first, a_b_2.first), max<double>(a_b_1.second, a_b_2.second));
}

double ReLU(double x) {
   return max<double>(0.0, x);
}

double ReLU6(double x) {
   return ReLU(x) - ReLU(x - 6);
}

double LeakyReLU(double x, double leakyreluc = 0.25) {
   return ReLU(x) - leakyreluc * ReLU(x);
}

double Hardsigmoid(double x) {
   return ReLU(x / 6 + 0.5) - ReLU(x / 6 - 0.5);
}

double Hardtanh(double x, double lb = -1, double ub = 1) {
   return ReLU(x - lb) - ReLU(x - ub);
}

double Softshrink(double x, double lambda = 0.5) {
   return ReLU(x - lambda) - ReLU(-x - lambda);
}

double Hardshrink(double x, double lambda = 0.5) {
   return Softshrink(x, lambda) + lambda * (x > lambda) - lambda * (x < -lambda);
}

double Threshold(double x, double threshold = 0.5, double value = 0.5) {
   return x > threshold ? x : value;
}

auto ReLU_bounds(const pair<double, double>& bounds_a) {
   return make_pair(ReLU(bounds_a.first), ReLU(bounds_a.second));
}

auto ReLU6_bounds(const pair<double, double>& bounds_a) {
   return make_pair(ReLU6(bounds_a.first), ReLU6(bounds_a.second));
}

auto LeakyReLU_bounds(const pair<double, double>& bounds_a, double leakyreluc = 0.25) {
   return make_pair(LeakyReLU(bounds_a.first, leakyreluc), LeakyReLU(bounds_a.second, leakyreluc));
}

auto Hardsigmoid_bounds(const pair<double, double>& bounds_a) {
   return make_pair(Hardsigmoid(bounds_a.first), Hardsigmoid(bounds_a.second));
}

auto Hardtanh_bounds(const pair<double, double>& bounds_a, double lb = -1, double ub = 1) {
   return make_pair(Hardtanh(bounds_a.first, lb, ub), Hardtanh(bounds_a.second, lb, ub));
}

auto Hardshrink_bounds(const pair<double, double>& bounds_a, double lambda = 0.5) {
   return make_pair(Hardshrink(bounds_a.first, lambda), Hardshrink(bounds_a.second, lambda));
}

auto Softshrink_bounds(const pair<double, double>& bounds_a, double lambda = 0.5) {
   return make_pair(Softshrink(bounds_a.first, lambda), Softshrink(bounds_a.second, lambda));
}

auto Threshold_bounds(const pair<double, double>& bounds_a, double threshold = -1, double value = 1) {
   if(bounds_a.first <= threshold && threshold < bounds_a.second) {
      return or_bounds({threshold, bounds_a.second}, value);
   }
   if(bounds_a.second <= threshold) {
      return make_pair(value, value);
   }
   return bounds_a;
}

void write_weights(const Tensor3Range<double> auto& weights, ostream& os) {
   int col = 0;
   for(const auto& weights : weights) {
      col = max<int>(col, ranges::distance(weights | views::join));
   }
   mat<string> data(int(ranges::distance(weights)) + 1, vec<string>(col + 3));
   data[0][1] = "d_0";
   data[0][2] = "d_1";
   ranges::move(views::iota(0, col) | views::transform([] (const auto& i) {
      return format("{}", i);
   }), data[0].begin() + 3);
   for(auto&& [i, data, weights] : views::zip(views::iota(0), data | views::drop(1), weights)) {
      data[0] = format("{}", i);
      data[1] = format("{}", ranges::distance(weights));
      data[2] = format("{}", ranges::distance(weights[0]));
      ranges::move(weights | views::join | views::transform([] (double w) {
         return format("{}", w);
      }), data.begin() + 3);
   }
   for(const auto& data : data) {
      for(const auto& data : data | views::join_with(',')) {
         os << data;
      }
      os << "\n";
   }
   os << flush;
}

double calculate_w(const RangeOf<double> auto& b, int exp) {
   double result = ldexp(-*ranges::begin(b), exp);
   for(const auto& [i, b] : b | views::enumerate | views::drop(1)) {
      result = fma(b, exp2(exp - i), result);
   }
   return result;
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
   string line, word; vec<optional<double>> L1w, L1a, L2w, L2a;
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
      optional<double> l1a, l1w, l2a, l2w, b;
      getline(line_stream, word, ','); stringstream(word) >> k;
      getline(line_stream, word, ','); stringstream(word) >> af;
      getline(line_stream, word, ','); stringstream(word) >> b;
      getline(line_stream, word, ','); stringstream(word) >> l1w;
      getline(line_stream, word, ','); stringstream(word) >> l1a;
      getline(line_stream, word, ','); stringstream(word) >> l2w;
      getline(line_stream, word, ','); stringstream(word) >> l2a;
      C.emplace_back(k);
      AF.emplace_back(af.value_or("None"));
      bias.emplace_back(b.value_or(1.0));
      L1w.emplace_back(l1w);
      L1a.emplace_back(l1a);
      L2w.emplace_back(l2w);
      L2a.emplace_back(l2a);
   }
   return make_tuple(C, AF, bias, L1w, L1a, L2w, L2a);
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
            a = clamp(b, min_value, max_value);
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
      {"tol0",              1}, {"reltol",            1}, {"relax_frac",        2}, {"slope",             1},
      {"err_prio",          1}, {"init",              0}, {"7z",                0}, {"min_constrs",       0},
      {"grb_con",           0}, {"local",             0}, {"sos",               0}, {"square",            0},
      {"min_bits",          1}, {"max_bits",          1}, {"min_exp",           1}, {"max_exp",           1},
      {"lazy",              1}, {"offtol",            1}, {"wpen",              1}, {"seed",              1},
      {"no_hint",           0}, {"no_index",          0}, {"no_header",         0},
      {"samples",           1}, {"no_shuffle",        0}, {"bound",             1}, {"sparsity",          2},
      {"cls_loss",          1}, {"reg_loss",          1}, {"lim_bits",          1}, {"neurons",           1},
      {"layers",            1}, {"activation",        1}, {"restrict",          0}, {"bias",              1},
      {"minmax",            0}, {"max_constrs",       0}, {"recall",            1}, {"log_slope",         1},
      {"use_bounds",        0}, {"use_asym",          0}
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

auto get_targets(const RangeOf<path> auto& files_path, bool ignore_header = true, bool ignore_index = true, const string& loss = "") {
   ten3<double> targets;
   vec<string> losses;
   ranges::move(files_path | views::transform([&ignore_header, &ignore_index] (const path& path) { 
      return read_matrix_from_csv<double>(fstream(path), ignore_header, ignore_index);
   }), back_inserter(targets));
   ranges::move(files_path | views::transform([&ignore_header, &ignore_index, &loss] (const path& path) { 
      string target_loss = loss;
      if(!ignore_header && !ignore_index) {
         fstream fs(path);
         string index_header;
         getline(fs, index_header, ',');
         if(index_header != "") {
            target_loss = index_header;
         }
      }
      return target_loss;
   }), back_inserter(losses));
   return make_tuple(targets, losses);
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
   int instances = ranges::distance(features);
   ten3<double> cls_tgt(instances), reg_tgt(instances);
   //Procesa los targets
   for(const auto& tgt : class_targets) {
      for(const auto& [t, tgt] : tgt | views::enumerate) {
         cls_tgt[t].emplace_back(tgt);
      }
   }
   for(const auto& tgt : regression_targets) {
      for(const auto& [t, tgt] : tgt | views::enumerate) {
         reg_tgt[t].emplace_back(tgt);
      }
   }
   //
   vec<int> indices(instances);
   ranges::iota(indices, 0);
   if(shuffle) {
      ranges::shuffle(indices, generator);
   }
   mat<double> train_features, test_features;
   ten3<double> train_cls_targets, train_reg_targets;
   ten3<double> test_cls_targets, test_reg_targets;
   ranges::copy(indices | views::take(samples) | views::transform([&features] (int i) {
      return features[i];
   }), back_inserter(train_features));
   ranges::copy(indices | views::drop(samples) | views::transform([&features] (int i) {
      return features[i];
   }), back_inserter(test_features));
   ranges::copy(indices | views::take(samples) | views::transform([&cls_tgt] (int i) {
      return cls_tgt[i];
   }), back_inserter(train_cls_targets));
   ranges::copy(indices | views::drop(samples) | views::transform([&cls_tgt] (int i) {
      return cls_tgt[i];
   }), back_inserter(test_cls_targets));
   ranges::copy(indices | views::take(samples) | views::transform([&reg_tgt] (int i) {
      return reg_tgt[i];
   }), back_inserter(train_reg_targets));
   ranges::copy(indices | views::drop(samples) | views::transform([&reg_tgt] (int i) {
      return reg_tgt[i];
   }), back_inserter(test_reg_targets));
   return make_tuple(train_features, train_cls_targets, train_reg_targets, test_features, test_cls_targets, test_reg_targets);
}

auto evaluate_activation(const RangeOf<double> auto& z, const string& type, const RangeOf<double> auto& params) {
   vec<double> a; 
   if(type == "ReLU") {
      for(const auto& z : z) {
         a.emplace_back(max<double>(z, 0));
      }
   }
   else if(type == "ReLU6") {
      for(const auto& z : z) {
         a.emplace_back(max<double>(z, 0) - max<double>(z - 6, 0));
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      vec<double> values(ranges::distance(z), 0.25);
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [z, leakyreluc] : views::zip(z, values)) {
         a.emplace_back(max<double>(z, 0) - values[0] * max<double>(z, 0));
      }
   }
   else if(type  == "Hardtanh") {
      array<double, 2> values{-1, 1};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& z : z) {
         a.emplace_back(values[0] + max<double>(z - values[0], 0) - max<double>(z - values[1], 0));
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& z : z) {
         a.emplace_back(max<double>(z / 6 + 0.5, 0) - max<double>(z / 6 - 0.5, 0));
      }
   }
   else if(type == "Hardshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& z : z) {
         a.emplace_back(z < values[0] ? 0 : z);
      }
   }
   else if(type == "Softshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& z : z) {
         a.emplace_back(max<double>(z - values[0], 0) - max<double>(z + values[0], 0));
      }
   }
   else if(type == "Threshold") {
      array<double, 2> values{0.5, 0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& z : z) {
         a.emplace_back(z < values[0] ? values[1] : z);
      }
   }
   else {
      ranges::move(z, back_inserter(a));
   } 
   return a;
}

auto evaluate(
   const Tensor3Range<double> auto& w,      const MatrixRange<double> auto& fx,
   const RangeOf<double> auto& bias,        const Tensor3Range<bool> auto& mask,
   const RangeOf<string> auto& af,          const MatrixRange<double> auto& params
) {
   mat<double> py;
   for(const auto& fx : fx) {
      vec<double> a;
      ranges::copy(fx, back_inserter(a));
      for(const auto& [w, bias, af, params, mask] : views::zip(w, bias, af, params, mask)) {
         vec<double> aw;
         a.emplace_back(bias);
         int j_max = -1;
         for(const auto& [w, a, mask] : views::zip(w, a, mask)) {
            for(const auto& [j, w, mask] : views::zip(views::iota(0), w, mask)) {
               if(j > j_max) {
                  aw.emplace_back(0);
                  j_max = j;
               }
               aw[j] += w * mask * a;
            }
         }
         a = evaluate_activation(aw, af, params);
      }
      py.emplace_back(a);
   }
   return py;
}

auto evaluate_class_loss(const RangeOf<double> auto& ty, const RangeOf<double> auto& py, const string& type, double eps = 0.0000001) {
   double loss = 0;
   if(type == "LogLoss" || type ==  "CrossEntropy" || type == "BCE" || "BCEWithLogits") {
      if(ranges::distance(ty) == 1) {
         double py_d = sigmoid(*ranges::begin(py));
         double ty_d = *ranges::begin(ty);
         loss += -ty_d * log(max(py_d, eps)) + (ty_d - 1) * log1p(max(-py_d, eps - 1));
      }
      else {
         double div = 0;
         for(const auto& py : py) {
            div += exp(py);
         }
         vec<double> softmax;
         for(const auto& py : py) {
            softmax.emplace_back(exp(py) / div);
         }
         for(const auto& [ty, py] : views::zip(ty, softmax)) {
            loss += -ty * log(max(py, eps));
         }
      }
   }
   else if(type == "KLDivergence") {
      if(ranges::distance(ty) == 1) {
         double py_d = sigmoid(*ranges::begin(py));
         double ty_d = *ranges::begin(ty);
         loss += ty_d * (log(max(ty_d, eps)) - log(max(py_d, eps)));
      }
      else {
         double div = 0;
         for(const auto& py : py) {
            div += exp(py);
         }
         vec<double> softmax;
         for(const auto& py : py) {
            softmax.emplace_back(exp(py) / div);
         }
         for(const auto& [ty, py] : views::zip(ty, softmax)) {
            loss += ty * (log(max(ty, eps)) - log(max(py, eps)));
         }
      }
   }
   return loss;
}

auto evaluate_class_accuracy(const RangeOf<double> auto& ty, const RangeOf<double> auto& py) {
   int hits = 0, fails = 0;
   if(ranges::distance(ty) == 1) {
      double ty_d = *ranges::begin(ty);
      double py_d = sigmoid(*ranges::begin(py));
      hits += py_d >= 0.5 && ty_d >= 0.5 || py_d <= 0.5 && ty_d <= 0.5;
      fails += py_d >= 0.5 && ty_d <= 0.5 || py_d <= 0.5 && ty_d >= 0.5;
   }
   else {
      double div = 0;
      for(const auto& py : py) {
         div += exp(py);
      }
      vec<double> softmax;
      for(const auto& py : py) {
         softmax.emplace_back(exp(py) / div);
      }
      vec<tuple<double, int>> py_tup, ty_tup;
      for(const auto& [py, i] : views::zip(softmax, views::iota(0))) {
         py_tup.emplace_back(make_tuple(py, i));
      }
      for(const auto& [ty, i] : views::zip(ty, views::iota(0))) {
         ty_tup.emplace_back(make_tuple(ty, i));
      }
      ranges::stable_sort(py_tup);
      ranges::stable_sort(ty_tup);
      for(const auto& [py_tup, ty_tup] : views::zip(py_tup, ty_tup)) {
         hits  += (get<1>(py_tup) == get<1>(ty_tup));
         fails += (get<1>(py_tup) != get<1>(ty_tup));
      }
   }
   return make_tuple(hits, fails);
}

auto evaluate_regression_loss(const RangeOf<double> auto& ty, const RangeOf<double> auto& py, const string& type) {
   double loss = 0;
   if(type == "L1" || type == "Absolute") {
      for(const auto& [ty, py] : views::zip(ty, py)) {
         loss += abs(ty - py);
      }
   }
   else if(type == "MSE" || type == "Square") {
      for(const auto& [ty, py] : views::zip(ty, py)) {
         loss += pow(ty - py, 2);
      }
   }
   return loss;
}

auto evaluate_accuracy(
   const Tensor3Range<double> auto& w,        const MatrixRange<double> auto& fx,
   const Tensor3Range<double> auto& class_ty, const RangeOf<double> auto& bias,
   const Tensor3Range<bool> auto& mask,       const RangeOf<string> auto& af,
   const MatrixRange<double> auto& params
) {
   double T = ranges::distance(fx);
   const auto& py = evaluate(w, fx, bias, mask, af, params);
   vec<tuple<double, double>> accuracies(ranges::distance(*ranges::begin(class_ty)), make_tuple(0.0, 0.0));
   for(const auto& [t, py, ty] : views::zip(views::iota(0), py, class_ty)) {
      int asize = 0;
      for(const auto& [i, ty] : ty | views::enumerate) {
         int size = ranges::distance(ty);
         if(size > 0) {
            auto [hits_a, fails_a] = accuracies[i];
            const auto& [hits, fails] = evaluate_class_accuracy(ty, py | views::drop(asize) | views::take(size));
            hits_a += hits / ranges::distance(ty) / T;
            fails_a += fails / ranges::distance(ty) / T;
            accuracies[i] = make_tuple(hits_a, fails_a);
            asize += size;
         }
      }
   }
   return accuracies;
}

auto evaluate_loss(
   const Tensor3Range<double> auto& w,             const MatrixRange<double> auto& fx,
   const Tensor3Range<double> auto& regression_ty, const Tensor3Range<double> auto& class_ty,
   const RangeOf<double> auto& bias,               const Tensor3Range<bool> auto& mask,
   const RangeOf<string> auto& af,                 const MatrixRange<double> auto& params,
   const RangeOf<string> auto& reg_loss,           const RangeOf<string> auto& class_loss
) {
   const auto& py = evaluate(w, fx, bias, mask, af, params);
   double T = ranges::distance(fx);
   vec<double> losses(ranges::distance(class_loss) + ranges::distance(reg_loss), 0);
   for(const auto& [t, py, cls_ty, reg_ty] : views::zip(views::iota(0), py, class_ty, regression_ty)) {
      int asize = 0;
      int loss_i = 0;
      for(const auto& [ty, loss] : views::zip(cls_ty, class_loss)) {
         int size = ranges::distance(ty);
         if(size > 0) {
            losses[loss_i] += evaluate_class_loss(ty, py | views::drop(asize) | views::take(size), loss) / T;
            asize += size;
         }
         ++loss_i;
      }
      for(const auto& [ty, loss] : views::zip(reg_ty, reg_loss)) {
         int size = ranges::distance(ty);
         if(size > 0) {
            losses[loss_i] = evaluate_regression_loss(ty, py | views::drop(asize) | views::take(size), loss) / T;
            asize += size;
         }
         ++loss_i;
      }
   }
   return losses;
}

class NNGRBCallback : public GRBCallback {
   public:
      NNGRBCallback(
         const path& file_path,                 const ten3<int>& exp,                  const ten3<optional<double>>& tw,
         const ten3<bool>& mask,                const mat<double>& train_features,     const mat<double>& test_features,
         const vec<double>& bias,               const vec<string>& af,                 const mat<double>& params,
         const ten3<double>& train_reg_targets, const ten3<double>& train_cls_targets, const ten3<double>& test_reg_targets,
         const ten3<double>& test_cls_targets,  const vec<string>& reg_loss,           const vec<string>& cls_loss
      ) :
         file_path(file_path), exp(exp), tw(tw), mask(mask), train_features(train_features), test_features(test_features), bias(bias),
         af(af), params(params), train_reg_targets(train_reg_targets), train_cls_targets(train_cls_targets),
         test_reg_targets(test_reg_targets), test_cls_targets(test_cls_targets),
         reg_loss(reg_loss), cls_loss(cls_loss)
      {
         ten3<double> w;
         for(const auto& tw : tw) {
            w.emplace_back(mat<double>());
            for(const auto& tw : tw) {
               w.back().emplace_back(vec<double>());
               for(const auto& tw : tw) {
                  w.back().back().emplace_back(tw.value_or(0));
               }
            }
         }
            cout << "Saving solution O...\n";
         fstream fo(file_path / format("weights_O.csv"), ios_base::out);
         if(fo.is_open()){
            write_weights(w, fo);
         }
         else {
            cout << "Saving weights failed\n";
         }
         fo = fstream(file_path / format("loss_O.log"), ios_base::out);
         if(fo.is_open()){
            write_loss(w, fo);
         }
         else {
            cout << "Saving loss failed\n";
         }
         fo = fstream(file_path / format("accuracy_O.log"), ios_base::out);
         if(fo.is_open()){
            write_accuracy(w, fo);
         }
         else {
            cout << "Saving accuracy failed\n";
         }
      }
      void write_accuracy(const Tensor3Range<double> auto& weights, ostream& os) {
         os << "Train accuracy\n";
         const auto& train_accuracies = evaluate_accuracy(weights, train_features, train_cls_targets, bias, mask, af, params);
         for(const auto& [i, tup] : train_accuracies | views::enumerate) {
            const auto& [hits, fails] = tup;
            os << "Classes " << i << ": hit% = " << hits / double(hits + fails) << " , fails% = " << fails / double(hits + fails) << "\n";
         }
         os << "Test accuracy\n";
         const auto& test_accuracies = evaluate_accuracy(weights, test_features, test_cls_targets, bias, mask, af, params);
         for(const auto& [i, tup] : test_accuracies | views::enumerate) {
            const auto& [hits, fails] = tup;
            os << "Classes " << i << ": hit% = " << hits / double(hits + fails) << " , fails% = " << fails / double(hits + fails) << "\n";
         }
      }
      void write_loss(const Tensor3Range<double> auto& weights, ostream& os) {
         os << "Train loss\n";
         const auto& train_losses = evaluate_loss(weights, train_features, train_reg_targets, train_cls_targets, bias, mask, af, params, reg_loss, cls_loss);
         for(const auto& [i, loss] : train_losses | views::enumerate) {
            os << "Loss " << i << " = " << loss << "\n";
         }
         os << "Test loss\n";
         const auto& test_losses = evaluate_loss(weights, test_features, test_reg_targets, test_cls_targets, bias, mask, af, params, reg_loss, cls_loss);
         for(const auto& [i, loss] : test_losses | views::enumerate) {
            os << "Loss " << i << " = " << loss << "\n";
         }
      }
      void set_binary(const ten4<GRBVar>& binary, const ten3<GRBVar>& iw) {
         this->binary = binary;
         this->iw = iw;
      }
   private:
      path file_path;
      ten4<GRBVar> binary;
      ten3<GRBVar> iw;
      ten3<optional<double>> tw;
      ten3<int> exp;
      ten3<bool> mask;
      ten3<double> train_cls_targets, train_reg_targets, test_cls_targets, test_reg_targets;
      mat<double> train_features, test_features, params;
      vec<double> bias;
      vec<string> af, cls_loss, reg_loss;
   protected:
      void callback() try {
         if(where == GRB_CB_MIPSOL) {
            int solcnt = getIntInfo(GRB_CB_MIPSOL_SOLCNT);
            ten3<double> sol(binary.size());
            for(auto&& [sol, binary, exp, iw, tw] : views::zip(sol, binary, exp, iw, tw)) {
               sol = mat<double>(binary.size());
               for(auto&& [sol, binary, exp, iw, tw] : views::zip(sol, binary, exp, iw, tw)) {
                  sol = vec<double>(binary.size());
                  for(auto&& [sol, binary, exp, iw, tw] : views::zip(sol, binary, exp, iw, tw)) {
                     vec<double> bin;
                     for(const auto& binary : binary) {
                        bin.emplace_back(getSolution(binary) < 0.5);
                     }
                     sol = calculate_w(bin, exp) + tw.value_or(0) * (getSolution(iw) < 0.5);
                  }
               }
            }
            cout << "Saving solution " << solcnt << "...\n";
            fstream fo(file_path / format("weights_{}.csv", solcnt), ios_base::out);
            if(fo.is_open()){
               write_weights(sol, fo);
            }
            else {
               cout << "Saving weights failed\n";
            }
            fo = fstream(file_path / format("loss_{}.log", solcnt), ios_base::out);
            if(fo.is_open()){
               write_loss(sol, fo);
            }
            else {
               cout << "Saving loss failed\n";
            }
            fo = fstream(file_path / format("accuracy_{}.log", solcnt), ios_base::out);
            if(fo.is_open()){
               write_accuracy(sol, fo);
            }
            else {
               cout << "Saving accuracy failed\n";
            }
         }
      }
      catch (const GRBException& ex) {
      }
      catch (...) {

      }
};

template<typename T>
auto decompose_w(const optional<T>& w, int size, int desired_exp = 0) {
   vec<T> bits_floor(size, 0.0), bits_ceil(size, 0.0);
   if(w) {
      int exp, sign = *w == 0.0 ? 0 : (1 - 2 * (*w < 0.0));
      T imantissa = frexp(*w, &exp) * exp2(size - 1 - desired_exp + exp);
      if(desired_exp >= exp) {
         int inf = floor(imantissa * -sign) * sign;
         int sup = ceil( imantissa * -sign) * sign;
         for(auto&& [bf, bc] : views::zip(bits_floor, bits_ceil) | views::reverse) {
            bf = inf & 1; bc = sup & 1;
            inf >>= 1;    sup >>= 1;
         }
      }
   }
   return make_tuple(bits_floor, bits_ceil);
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
      set_constr(model, var_u + expr == ub, name, lazy);
   }
}

GRBLinExpr gen_sum_expr(const GRBLinExprRange auto& X) {
   return accumulate(ranges::begin(X), ranges::end(X), GRBLinExpr());
}

GRBLinExpr gen_mean_expr(const GRBLinExprRange auto& X) {
   return gen_sum_expr(X) / max<double>(1, ranges::distance(X));
}

auto gen_var(
   GRBModel& model,           const GRBLinExpr& expr,   const string& name,
   double lb = -GRB_INFINITY, double ub = GRB_INFINITY, char vtype = GRB_CONTINUOUS,
   int lazy = 0
) {
   if(is_single_expr(expr)) {
      return expr.getVar(0);
   }
   GRBVar var = model.addVar(lb, ub, 0, vtype, name);
   set_constr(model, expr == var, name, lazy);
   return var;
}

auto gen_abs_vars(
   GRBModel& model, const GRBLinExpr& x, const string& name,
   int lazy = 0,    int priority = 0,    const optional<pair<double, double>>& bounds_x = {}
) {
   const auto&  plus_name = format("{}_plus",   name);
   const auto& minus_name = format("{}_minus",  name);
   const auto&  on_name = format("{}_on",  name);
   const auto& off_name = format("{}_off", name);
   GRBVar var_abs, a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or", name));
   set_priority(a_or_b, priority);
   if(bounds_x) {
      const auto& bounds_abs = abs_bounds(*bounds_x);
      var_abs = model.addVar(bounds_abs.first, bounds_abs.second, 0, GRB_CONTINUOUS, name);
      double plus_ub  = bounds_abs.second - bounds_x->first;
      double minus_ub = bounds_abs.second + bounds_x->second;
      set_constr(model, plus_ub  *      a_or_b  + x >= var_abs, off_name, lazy);
      set_constr(model, minus_ub * (1 - a_or_b) - x >= var_abs,  on_name, lazy);
   }
   else {
      var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
      model.addGenConstrIndicator(a_or_b, 0,  x >= var_abs, off_name);
      model.addGenConstrIndicator(a_or_b, 1, -x >= var_abs,  on_name);
   }
   set_constr(model, var_abs >=  x, plus_name,  lazy);
   set_constr(model, var_abs >= -x, minus_name, lazy);
   return make_tuple(var_abs, a_or_b);
}

auto gen_abs_var(
   GRBModel& model, const GRBLinExpr& x, const string& name,
   int lazy = 0,    int priority = 0,    const optional<pair<double, double>>& bounds_x = {}
) {
   const auto& [var_abs, a_or_b] = gen_abs_vars(model, x, name, lazy, priority, bounds_x);
   return var_abs;
}

auto gen_abs_var(
   GRBModel& model, const GRBLinExpr& x, const string& name,
   int lazy = 0,    int priority = 0,    bool use_grb = false,
   const optional<pair<double, double>>& bounds_x = {}
) {
   GRBVar var_abs;
   if(use_grb) {
      GRBVar new_x;
      if(bounds_x) {
         const auto& bounds_abs = abs_bounds(*bounds_x);
         new_x = gen_var(
            model, x, format("{}_in", name),
            bounds_x->first, bounds_x->second, GRB_CONTINUOUS, lazy
         );
         var_abs = model.addVar(bounds_abs.first, bounds_abs.second, 0, GRB_CONTINUOUS, name);
      }
      else {
         new_x = gen_var(
            model, x, format("{}_in", name),
            -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, lazy
         );
         var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
      }
      model.addGenConstrAbs(var_abs, new_x, name);
   }
   else {
      var_abs = gen_abs_var(model, x, name, lazy, priority, bounds_x);
   }
   return var_abs;
}

GRBLinExpr gen_abs_expr(
   GRBModel& model, const GRBLinExpr& x, const string& name,
   int lazy = 0,    int priority = 0,    bool use_grb = false,
   const optional<pair<double, double>>& bounds_x = {}
) {
   return gen_abs_var(model, x, name, lazy, priority, use_grb, bounds_x);
}

auto gen_ReLU_vars(
   GRBModel& model, const GRBLinExpr& z, const string& name,
   int lazy = 0,    int priority = 0,    bool use_sos = false,
   const optional<pair<double, double>>& bounds_z = {}
) {
   const auto& off_name = format("{}_off", name);
   const auto& on_name  = format("{}_on",  name);
   GRBVar var, on_0 = model.addVar(0, 1, 0, GRB_BINARY, format("{}_0", name));
   set_priority(on_0, priority);
   if(bounds_z) {
      const auto& bounds_relu = ReLU_bounds(*bounds_z);
      var = model.addVar(bounds_relu.first, bounds_relu.second, 0, GRB_CONTINUOUS, name);
      double max_bounds = bounds_relu.second - bounds_z->first;
      set_constr(model, var <=        z + max_bounds * on_0,  off_name, lazy);
      set_constr(model, var <= bounds_z->second * (1 - on_0), on_name,  lazy);
   }
   else {
      var = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
      model.addGenConstrIndicator(on_0, 0, var <= z, off_name);
      if(use_sos) {
         array<GRBVar, 2> vars_z{on_0, var}; array<double, 2> weights_z{0, 1};
         model.addSOS(vars_z.data(), weights_z.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(on_0, 1, var <= 0, on_name);
      }
   }
   set_constr(model, var >= z, name, lazy);
   return make_tuple(var, on_0);
}

auto gen_ReLU_var(
   GRBModel& model, const GRBLinExpr& z, const string& name,
   int lazy = 0,    int priority = 0,    bool use_sos = false,
   const optional<pair<double, double>>& bounds_z = {}
) {
   const auto& [var, on_0] = gen_ReLU_vars(model, z, name, lazy, priority, use_sos, bounds_z);
   return var;
}

auto gen_ReLU_var(
   GRBModel& model,  const GRBLinExpr& z,  const string& name,   int lazy = 0,
   int priority = 0, bool use_grb = false, bool use_sos = false, const optional<pair<double, double>>& bounds_z = {}
) {
   GRBVar relu_var;
   if(use_grb) {
      GRBVar var, new_z;
      if(bounds_z) {
         const auto& bounds_relu = ReLU_bounds(*bounds_z);
         var = model.addVar(bounds_relu.first, bounds_relu.second, 0, GRB_CONTINUOUS, name);
         new_z = gen_var(
            model, z, format("{}_in", name),
            bounds_z->first, bounds_z->second, GRB_CONTINUOUS, lazy
         );
      }
      else {
         var = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
         new_z = gen_var(
            model, z, format("{}_in", name),
            -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, lazy
         );
      } 
      model.addGenConstrMax(var, &new_z, 1, 0, name);
      relu_var = var;
   }
   else {
      const auto& [var, on_z] = gen_ReLU_vars(model, z, name, lazy, priority, use_sos, bounds_z);
      relu_var = var;
   }
   return relu_var;
}

GRBLinExpr gen_ReLU_expr(
   GRBModel& model,  const GRBLinExpr& z,  const string& name,   int lazy = 0,
   int priority = 0, bool use_grb = false, bool use_sos = false, const optional<pair<double, double>>& bounds_z = {}
) {
   return gen_ReLU_var(model, z, name, lazy, priority, use_grb, use_sos, bounds_z);
}

GRBLinExpr gen_Hardtanh_expr(
   GRBModel& model,  const GRBLinExpr& z,  const string& name,
   double lb = -1,   double ub = 1,        int lazy = 0,
   int priority = 0, bool use_grb = false, bool use_sos = false,
   const optional<pair<double, double>>& bounds_z = {}
) {
   const auto& zl = gen_ReLU_expr(model, z - lb, format("{}_zl", name), lazy, priority, use_grb, use_sos, bounds_z);
   const auto& zu = gen_ReLU_expr(model, z - ub, format("{}_zu", name), lazy, priority, use_grb, use_sos, bounds_z);
   return lb + zl - zu;
}

GRBLinExpr gen_Hardsigmoid_expr(
   GRBModel& model,  const GRBLinExpr& z,  const string& name,   int lazy = 0,
   int priority = 0, bool use_grb = false, bool use_sos = false, const optional<pair<double, double>>& bounds_z = {}
) {
   const auto& new_z  = z / 6 + 0.5;
   const auto& z0 = gen_ReLU_expr(model, new_z,     format("{}_z0", name), lazy, priority, use_grb, use_sos, bounds_z);
   const auto& z1 = gen_ReLU_expr(model, new_z - 1, format("{}_z1", name), lazy, priority, use_grb, use_sos, bounds_z);
   return z0 - z1;
}

GRBLinExpr gen_ReLU6_expr(
   GRBModel& model,  const GRBLinExpr& z,  const string& name,   int lazy = 0,
   int priority = 0, bool use_grb = false, bool use_sos = false, const optional<pair<double, double>>& bound_z = {}
) {
   const auto& z0 = gen_ReLU_expr(model, z,     format("{}_z0", name), lazy, priority, use_grb, use_sos, bound_z);
   const auto& z6 = gen_ReLU_expr(model, z - 6, format("{}_z6", name), lazy, priority, use_grb, use_sos, bound_z);
   return z0 - z6;
}

GRBLinExpr gen_Hardshrink_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,   double lambda = 0.5, 
   int lazy = 0,    int priority = 0,     bool use_sos = false, const optional<pair<double, double>>& bound_z = {}
) {
   const auto& [upper, on_upper0] = gen_ReLU_vars(model,   z - lambda,  name, lazy, priority, use_sos, bound_z);
   const auto& [lower, on_lower0] = gen_ReLU_vars(model, -(z + lambda), name, lazy, priority, use_sos, bound_z);
   return upper + lambda * (1 - on_upper0) - lower - lambda * (1 - on_lower0);
}

GRBLinExpr gen_Softshrink_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,   double lambda = 0.5,
   int lazy = 0,    int priority = 0,     bool use_grb = false, bool use_sos = false,
   const optional<pair<double, double>>& bounds_z = {}
) {
   const auto& zp = gen_ReLU_expr(model,   z - lambda,  format("{}_plus",  name), lazy, priority, use_grb, use_sos, bounds_z);
   const auto& zm = gen_ReLU_expr(model, -(z + lambda), format("{}_minus", name), lazy, priority, use_grb, use_sos, bounds_z);
   return zp - zm;
}

GRBLinExpr gen_Threshold_expr(
   GRBModel& model,      const GRBLinExpr& z,  const string& name,     int lazy = 0,
   int priority = 0,     bool use_sos = false, double threshold = 0.5, double value = 0.5,
   const optional<pair<double, double>>& bounds_z = {}
) {
   const auto& [zt, on_0] = gen_ReLU_vars(model, z - threshold, name, lazy, priority, use_sos, bounds_z);
   return zt + value + (threshold - value) * (1 - on_0);
}

GRBLinExpr gen_LeakyReLU_expr(
   GRBModel& model, const GRBLinExpr& z,  const string& name,   double reluc = 0.25,
   int lazy = 0,    int priority = 0,     bool use_grb = false, const optional<pair<double, double>>& bounds_z = {}
) {
   const auto& z_abs = gen_abs_expr(model, z, name, lazy, priority, use_grb, bounds_z);
   const auto& min_z0 = (z - z_abs) / 2;
   const auto& max_z0 = (z + z_abs) / 2;
   return max_z0 + reluc * min_z0;
}

auto activation_bounds(const string& type, const RangeOf<optional<pair<double, double>>> auto& bounds_z, const RangeOf<double> auto& params) {
   vec<optional<pair<double, double>>> a;
   if(type == "ReLU") {
      for(const auto& bounds_z : bounds_z) {
         if(bounds_z) {
            a.emplace_back(ReLU_bounds(*bounds_z));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else if(type == "ReLU6") {
      for(const auto& bounds_z : bounds_z) {
         if(bounds_z) {
            a.emplace_back(ReLU6_bounds(*bounds_z));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      vec<double> values(ranges::distance(bounds_z), 0.25);
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [bounds_z, leakyreluc] : views::zip(bounds_z, values)) {
         if(bounds_z) {
            a.emplace_back(LeakyReLU_bounds(*bounds_z, leakyreluc));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else if(type  == "Hardtanh") {
      array<double, 2> values{-1, 1};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& bounds_z : bounds_z) {
         if(bounds_z) {
            a.emplace_back(Hardtanh_bounds(*bounds_z, values[0], values[1]));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& bounds_z : bounds_z) {
         if(bounds_z) {
            a.emplace_back(Hardsigmoid_bounds(*bounds_z));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else if(type == "Hardshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& bounds_z : bounds_z) {
         if(bounds_z) {
            a.emplace_back(Hardshrink_bounds(*bounds_z, values[0]));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else if(type == "Softshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& bounds_z : bounds_z) {
         if(bounds_z) {
            a.emplace_back(Softshrink_bounds(*bounds_z, values[0]));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else if(type == "Threshold") {
      array<double, 2> values{0.5, 0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& bounds_z : bounds_z) {
         if(bounds_z) {
            a.emplace_back(Threshold_bounds(*bounds_z, values[0], values[1]));
         }
         else {
            a.emplace_back(bounds_z);
         }
      }
   }
   else {
      ranges::move(bounds_z, back_inserter(a));
   } 
   return a;
}

auto gen_activation_exprs(
   GRBModel& model,    const string& type,                 const GRBLinExprRange auto& z,
   const RangeOf<optional<pair<double, double>>> auto& bounds_z,
   const string& name, const RangeOf<double> auto& params, int lazy = 0,
   int priority = 0,   bool use_grb = false,               bool use_sos = false
) {
   vec<GRBLinExpr> a; string new_name;
   for(unsigned char c : type) {
      new_name.push_back(tolower(c));
   }
   new_name = format("{}_{}", name, new_name);
   if(type == "ReLU") {
      for(const auto& [j, z, bounds_z] : views::zip(views::iota(0), z, bounds_z)) {
         a.emplace_back(gen_ReLU_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, priority, use_grb, use_sos, bounds_z
         ));
      }
   }
   else if(type == "ReLU6") {
      for(const auto& [j, z, bounds_z] : views::zip(views::iota(0), z, bounds_z)) {
         a.emplace_back(gen_ReLU6_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, priority, use_grb, use_sos, bounds_z
         ));
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      vec<double> values(ranges::distance(z), 0.25);
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z, bounds_z, leakyreluc] : views::zip(views::iota(0), z, bounds_z, values)) {
         a.emplace_back(gen_LeakyReLU_expr(
            model, z,
            format("{}_{}", new_name, j),
            leakyreluc, lazy, priority, use_grb, bounds_z
         ));
      }
   }
   else if(type  == "Hardtanh") {
      array<double, 2> values{-1, 1};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z, bounds_z] : views::zip(views::iota(0), z, bounds_z)) {
         a.emplace_back(gen_Hardtanh_expr(
            model, z,
            format("{}_{}", new_name, j),
            values[0], values[1], lazy, priority, use_grb,
            use_sos, bounds_z
         ));
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& [j, z, bounds_z] : views::zip(views::iota(0), z, bounds_z)) {
         a.emplace_back(gen_Hardsigmoid_expr(
            model, z,
            format("{}_{}", new_name, j),
            lazy, priority, use_grb, use_sos, bounds_z
         ));
      }
   }
   else if(type == "Hardshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z, bounds_z] : views::zip(views::iota(0), z, bounds_z)) {
         a.emplace_back(gen_Hardshrink_expr(
            model, z,
            format("{}_{}", new_name, j),
            values[0], lazy, priority, use_sos, bounds_z
         ));
      }
   }
   else if(type == "Softshrink") {
      array<double, 1> values{0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z, bounds_z] : views::zip(views::iota(0), z, bounds_z)) {
         a.emplace_back(gen_Softshrink_expr(
            model, z,
            format("{}_{}", new_name, j),
            values[0], lazy, priority, use_grb, use_sos, bounds_z
         ));
      }
   }
   else if(type == "Threshold") {
      array<double, 2> values{0.5, 0.5};
      for(auto&& [v, p] : views::zip(values, params)) {
         v = p;
      }
      for(const auto& [j, z, bounds_z] : views::zip(views::iota(0), z, bounds_z)) {
         a.emplace_back(gen_Threshold_expr(
            model, z,
            format("{}_{}", new_name, j),
            values[0], values[1], lazy, priority, use_sos, bounds_z
         ));
      }
   }
   else {
      ranges::move(z, back_inserter(a));
   } 
   return a;
}

auto gen_sigmoid_error_var(
   GRBModel& model,     const GRBLinExpr& y,  const string& name, const string& loss, double ty,
   double cls_w,        double eps = 2.06e-9, double tol = 0.1,   int priority = 0,   bool restrict = false,
   bool minmax = false, int lazy = 0,         const optional<GRBVar>& is_constr = {}, const optional<double>& recall = {},
   const optional<pair<double, double>>& bounds_y = {},                               const optional<double>& slope = {}
) {
   int loss_type = 0;
   if(loss == "BCE" || loss == "CrossEntropy" || loss == "BCEWithLogits" || loss == "LogLoss" || loss == "NLL") {
      loss_type = 0;
   }
   else if(loss == "KLDivergence") {
      loss_type = 1;
   }
   const auto& ic_name       = format("{}_isconstr", name);
   const auto& restrict_name = format("{}_restrict", name);
   const auto& minmax_name   = format("{}_minmax",   name);
   const auto& loss_name     = format("{}_loss",     name);
   double size_d             = 2;
   double ub_softmax         = log(size_d);
   double ty_d               = ty;
   double cls_weight         = cls_w;
   double rec                = recall.value_or(1 / (cls_weight - 1));
   GRBLinExpr yi             = y;
   GRBVar loss_cls           = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, loss_name);
   vec<GRBLinExpr> exprs, minmax_exprs;
   exprs.emplace_back(loss_cls);
   double loss_sum_ub = 0, loss_sum_lb = 0, minmax_sum_ub = 0;
   vec<GRBLinExpr> pos_exprs, neg_exprs;
   double slope_d = slope.value_or(eps);
   double slope_half    = tan(pi_v<double> / 8);
   double lsm_half_d    = -log1p(-slope_half);
   double lsm_x_half_d  = log(slope_half) + lsm_half_d;
   double lsm_eps_d     = -log1p(-slope_d);
   double lsm_x_eps_d   = log(slope_d) + lsm_eps_d;
   pos_exprs.emplace_back(              -yi);
   pos_exprs.emplace_back(slope_half * (-yi - lsm_x_half_d) + lsm_half_d);
   pos_exprs.emplace_back(   slope_d * (-yi - lsm_x_eps_d ) + lsm_eps_d );
   neg_exprs.emplace_back(               yi);
   neg_exprs.emplace_back(slope_half * ( yi - lsm_x_half_d) + lsm_half_d);
   neg_exprs.emplace_back(   slope_d * ( yi - lsm_x_eps_d ) + lsm_eps_d );
   GRBVar pos = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_pos", loss_name));
   GRBVar neg = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_neg", loss_name));
   double loss_i_lb, loss_i_ub;
   pair<double, double> bounds_yi;
   if(bounds_y) {
      vec<double> bounds_pos, bounds_neg;
      bounds_pos.emplace_back(              -bounds_y->first);
      bounds_pos.emplace_back(slope_half * (-bounds_y->first  - lsm_x_half_d) + lsm_half_d);
      bounds_pos.emplace_back(   slope_d * (-bounds_y->first  - lsm_x_eps_d ) + lsm_eps_d );
      bounds_neg.emplace_back(               bounds_y->second);
      bounds_neg.emplace_back(slope_half * ( bounds_y->second - lsm_x_half_d) + lsm_half_d);
      bounds_neg.emplace_back(   slope_d * ( bounds_y->second - lsm_x_eps_d ) + lsm_eps_d );
      double pos_ub = 0, neg_ub = 0;
      for(const auto& [bounds_pos, bounds_neg] : views::zip(bounds_pos, bounds_neg)) {
         pos_ub = max<double>(pos_ub, bounds_pos);
         neg_ub = max<double>(neg_ub, bounds_neg);
      }
      set_ub(pos, pos_ub);
      set_ub(neg, neg_ub);
      if(loss_type == 0) {
         loss_i_ub = ( rec * ty_d * pos_ub + (1 - ty_d) * neg_ub ) * cls_weight;
      }
      else if(loss_type == 1) {
         loss_i_ub = ty_d * ( log(max<double>(ty_d, eps)) + pos_ub );
      }
      set_ub(loss_cls, loss_i_ub);
      loss_sum_ub += loss_i_ub;
   }
   for(const auto& [i, pos_expr, neg_expr] : views::zip(views::iota(0), pos_exprs, neg_exprs)) {
      set_constr(model, pos >= pos_expr, format("{}_{}_pos", loss_name, i), lazy);
      set_constr(model, neg >= neg_expr, format("{}_{}_neg", loss_name, i), lazy);
   }
   if(loss_type == 0) {
      loss_i_lb = 0.0;
   }
   else if(loss_type == 1) {
      loss_i_lb = ty_d * ( log(max<double>(ty_d, eps))  );
   }
   set_lb(loss_cls, loss_i_lb);
   loss_sum_lb += loss_i_lb;
   if(loss_type == 0) {
      set_constr(model, loss_cls == ( rec * ty_d * pos + (1 - ty_d) * neg ) * cls_weight, loss_name, lazy);
   }
   else if(loss_type == 1) {
      set_constr(model, loss_cls == ty_d * ( log(max<double>(ty_d, eps)) + pos ), loss_name, lazy);
   }
   if(is_constr) {
      double ub = clamp<double>(ty_d + tol, 0, 1), lb = clamp<double>(ty_d - tol, 0, 1);
      double loss_ub = GRB_INFINITY;
      for(const auto& ub : array{lb, ub}) {
         if(0.0 + eps < ub && ub < 1.0 - eps ) {
            if(loss_type == 0) {
               loss_ub = min<double>(loss_ub, binary_cross_entropy_loss(ub, ty_d, cls_weight, rec, eps));
            }
            else if(loss_type == 1) {
               loss_ub = min<double>(loss_ub, kldivergence_loss(ub, ty_d, eps));
            }
         }
      }
      if(bounds_y) {
         double max_bounds = loss_i_ub - loss_ub;
         set_constr(model, loss_cls <= loss_ub + max_bounds * (1 - *is_constr), ic_name, lazy);
      }
      else {
         model.addGenConstrIndicator(*is_constr, 1, loss_cls <= loss_ub, ic_name);
      }
      if(minmax) {
         minmax_sum_ub += loss_ub;
         GRBVar loss_d = model.addVar(0, loss_ub, 0, GRB_CONTINUOUS, minmax_name);
         minmax_exprs.emplace_back(loss_d);
         if(loss_type == 0) {
            vec<GRBLinExpr> exprs;
            exprs.emplace_back(loss_d - loss_ub + ( (ty_d * (rec - 1) + 1) * yi - ty_d * rec * yi ) * cls_weight);
            exprs.emplace_back(loss_d - loss_ub + (                             - ty_d * rec * yi ) * cls_weight);
            vec<GRBVar> binaries;
            for(const auto& bi : views::iota(0, int(exprs.size()))) {
               auto var = model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi));
               binaries.emplace_back(var);
               set_priority(var, priority);
            }
            if(bounds_y) {
               vec<double> max_bounds;
               max_bounds.emplace_back(-loss_ub + ( (ty_d * (rec - 1) + 1) * bounds_yi.first - ty_d * rec * bounds_yi.second ) * cls_weight);
               max_bounds.emplace_back(-loss_ub + (                                          - ty_d * rec * bounds_yi.second ) * cls_weight);
               for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                  set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
               }
            }
            else {
               for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                  model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
               }
            }
            set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
         }
         else if (loss_type == 1) {
            vec<GRBLinExpr> exprs;
            exprs.emplace_back(loss_d - loss_ub + ty_d * (log(max<double>(ty_d, eps)) - yi));
            exprs.emplace_back(loss_d - loss_ub + ty_d *  log(max<double>(ty_d, eps)));
            vec<GRBVar> binaries;
            for(const auto& bi : views::iota(0, int(exprs.size()))) {
               auto var = model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi));
               binaries.emplace_back(var);
               set_priority(var, priority);
            }
            if(bounds_y) {
               vec<double> max_bounds;
               max_bounds.emplace_back(-loss_ub + ty_d * (log(max<double>(ty_d, eps)) - bounds_yi.second));
               max_bounds.emplace_back(-loss_ub + ty_d *  log(max<double>(ty_d, eps)));
               for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                  set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
               }
            }
            else {
               for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                  model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
               }
            }
            set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
         }
      }
   }
   GRBVar obj;
   if(is_constr) {
      double obj_ub = bounds_y ? loss_sum_ub - loss_sum_lb : GRB_INFINITY;
      obj = model.addVar(0, obj_ub, 0, GRB_CONTINUOUS, name);
      if(minmax) {
         set_constr(model, obj >= gen_sum_expr(minmax_exprs) - minmax_sum_ub * (1 - *is_constr), minmax_name, lazy);
         obj_ub = max<double>(obj_ub, minmax_sum_ub);
         set_ub(obj, obj_ub);
      }
      if(bounds_y) {
         if(restrict) {
            double max_bound = -size_d * ub_softmax + loss_sum_lb + obj_ub;
            set_constr(model, obj <= size_d * ub_softmax - loss_sum_lb + max_bound * (*is_constr), restrict_name, lazy);
         }
         double max_bound = -loss_sum_ub + loss_sum_lb;
         set_constr(model, obj >= gen_sum_expr(exprs) - loss_sum_lb + max_bound * (*is_constr), name, lazy);
      }
      else {
         if(restrict) {
            model.addGenConstrIndicator(*is_constr, 0, obj <= size_d * ub_softmax - loss_sum_lb, restrict_name);
         }
         model.addGenConstrIndicator(*is_constr, 0, obj >= gen_sum_expr(exprs) - loss_sum_lb, name);
      }
   }
   else {
      double obj_ub = bounds_y ? loss_sum_ub - loss_sum_lb : GRB_INFINITY;
      if(restrict) {
         obj_ub = min<double>(obj_ub, size_d * ub_softmax - loss_sum_lb);
      }
      obj = model.addVar(0, obj_ub, 0, GRB_CONTINUOUS, name);
      set_constr(model, obj >= gen_sum_expr(exprs) - loss_sum_lb, name, lazy);
   }
   return obj;
}

auto gen_softmax_error_var(
   GRBModel& model,      const GRBLinExprRange auto& y, const RangeOf<optional<pair<double, double>>> auto& bounds_y,
   const string& name,   const string& loss,            const RangeOf<double> auto& ty,         const RangeOf<double> auto& cls_w,
   double eps = 2.06e-9, double tol = 0.1,              int priority = 0,                       bool restrict = false,
   bool minmax = false,  int lazy = 0,                  const optional<GRBVar>& is_constr = {}, const optional<double>& slope = {}
) {
   int loss_type = 0;
   if(loss == "BCE" || loss == "CrossEntropy" || loss == "BCEWithLogits" || loss == "LogLoss" || loss == "NLL") {
      loss_type = 0;
   }
   else if(loss == "KLDivergence") {
      loss_type = 1;
   }
   bool use_bounds = ranges::all_of(bounds_y, [](const auto& opt){ return opt.has_value(); });
   int size                  = ranges::distance(ty);
   const auto& ic_name       = format("{}_isconstr", name);
   const auto& restrict_name = format("{}_restrict", name);
   const auto& minmax_name   = format("{}_minmax",   name);
   const auto& loss_name     = format("{}_loss",     name);
   vec<GRBLinExpr> exprs, minmax_exprs;
   double size_d = max<double>(2, size);
   double ub_softmax = log(size_d);
   double loss_sum_ub = 0, loss_sum_lb = 0, minmax_sum_ub = 0;
   double slope_d = slope.value_or(eps);
   vec<double> lsm_c_eq, lsm_eq, lsm_c, lsm, slopes_comp, slopes;
   for(const auto& slope : array{slope_d, tan(pi_v<double> / 8)}) {
      slopes.emplace_back(slope);
      slopes_comp.emplace_back((1 - slope) / size_d);
      lsm_eq.emplace_back(                      - log1p(-slope));
      lsm_c_eq.emplace_back(log(slope / size_d) - log1p(-slope));
      lsm.emplace_back(             - log1p(-slope) + log(size_d));
      lsm_c.emplace_back(log(slope) - log1p(-slope) + log(size_d));
   }
   for(const auto& [cls, ty_d, yi, bounds_yi, cls_w] : views::zip(views::iota(0), ty, y, bounds_y, cls_w)) {
      const auto& minmax_name = format("{}_minmax_{}",   name, cls);
      const auto& loss_name   = format("{}_loss_{}",     name, cls);
      const auto& ic_name     = format("{}_isconstr_{}", name, cls);
      double cls_weight       = cls_w;
      GRBVar loss_cls = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, loss_name);
      double loss_i_ub, loss_i_lb;
      GRBVar log_softmax = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_logsoftmax", loss_name));
      if(use_bounds) {
         double lsm_ub = 0.0;
         vec<double> log_softmax_ub;
         for(const auto& bounds_yj : bounds_y) {
            log_softmax_ub.emplace_back(bounds_yj->second - bounds_yi->first);
         }
         for(const auto& [slope, slope_comp, lsm_c, lsm, lsm_c_eq, lsm_eq] : views::zip(slopes, slopes_comp, lsm_c, lsm, lsm_c_eq, lsm_eq)) {
            vec<double> c_ub, c_comp_ub, eq_ub;
            for(const auto& bounds_yj : bounds_y) {
               c_ub.emplace_back(slope * (bounds_yj->second - bounds_yi->first - lsm_c));
               c_comp_ub.emplace_back(slope_comp * (bounds_yj->second - bounds_yi->first));
               eq_ub.emplace_back(slope / size_d * (bounds_yj->second - bounds_yi->first - lsm_c_eq));
            }
            double c_sum_ub  = accumulate(c_comp_ub.begin(), c_comp_ub.end(), lsm);
            double eq_sum_ub = accumulate(eq_ub.begin(), eq_ub.end(), lsm_eq);
            for(const auto& [c_ub, c_comp_ub] : views::zip(c_ub, c_comp_ub)) {
               log_softmax_ub.emplace_back(c_sum_ub - c_comp_ub + c_ub);
            }
            log_softmax_ub.emplace_back(eq_sum_ub);
         }
         for(const auto& ub : log_softmax_ub) {
            lsm_ub = max<double>(lsm_ub, ub);
         }
         set_ub(log_softmax, lsm_ub);
         if(loss_type == 0) { 
            loss_i_ub = ty_d * cls_weight * lsm_ub;
         }
         else if(loss_type == 1) {
            loss_i_ub = ty_d * (log(max<double>(ty_d, eps)) + lsm_ub);
         }
         set_ub(loss_cls, loss_i_ub);
      }
      vec<GRBLinExpr> log_softmax_exprs;
      for(const auto& yj : y) {
         log_softmax_exprs.emplace_back(yj - yi);
      }
      for(const auto& [slope, slope_comp, lsm_c, lsm, lsm_c_eq, lsm_eq] : views::zip(slopes, slopes_comp, lsm_c, lsm, lsm_c_eq, lsm_eq)) {
         vec<GRBLinExpr> c_exprs, c_comp_exprs, eq_exprs;
         for(const auto& yj : y) {
            c_exprs.emplace_back(slope * (yj - yi - lsm_c));
            c_comp_exprs.emplace_back(slope_comp * (yj - yi));
            eq_exprs.emplace_back(slope / size_d * (yj - yi - lsm_c_eq));
         }
         GRBLinExpr c_sum_expr  = gen_sum_expr(c_comp_exprs) + lsm;
         GRBLinExpr eq_sum_expr = gen_sum_expr(eq_exprs) + lsm_eq;
         for(const auto& [c_expr, c_comp_expr] : views::zip(c_exprs, c_comp_exprs)) {
            log_softmax_exprs.emplace_back(c_sum_expr - c_comp_expr + c_expr);
         }
         log_softmax_exprs.emplace_back(eq_sum_expr);
      }
      for(const auto& [i, expr] : log_softmax_exprs | views::enumerate) {
         set_constr(model, log_softmax >= expr, format("{}_{}_logsoftmax", loss_name, i), lazy);
      }
      if(loss_type == 0) { 
         loss_i_lb = 0.0;
      }
      else if(loss_type == 1) {
         loss_i_lb = ty_d * (log(max<double>(ty_d, eps)));
      }
      set_lb(loss_cls, loss_i_lb);
      if(loss_type == 0) {
         set_constr(model, loss_cls == cls_weight * ty_d * log_softmax, loss_name, lazy);
      }
      else if(loss_type == 1) {
         set_constr(model, loss_cls == ty_d * (log(max<double>(ty_d, eps)) + log_softmax), loss_name, lazy);
      }
      exprs.emplace_back(loss_cls);
      if(is_constr) {
         double ub = clamp<double>(ty_d + tol, 0, 1), lb = clamp<double>(ty_d - tol, 0, 1);
         double loss_ub = GRB_INFINITY;
         for(const auto& ub : array{lb, ub}) {
            if(loss_type == 0) {
               loss_ub = min<double>(loss_ub, cross_entropy_loss(ub, ty_d, cls_weight, eps));
            }
            else if(loss_type == 1) {
               loss_ub = min<double>(loss_ub, kldivergence_loss(ub, ty_d, eps));
            }
         }
         if(use_bounds) {
            double max_bounds = loss_i_ub - loss_ub;
            set_constr(model, loss_cls <= loss_ub + max_bounds * (1 - *is_constr), ic_name, lazy);
         }
         else {
            model.addGenConstrIndicator(*is_constr, 1, loss_cls <= loss_ub, ic_name);
         }
         if(minmax) {
            minmax_sum_ub += loss_ub;
            GRBVar loss_d = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, minmax_name);
            minmax_exprs.emplace_back(loss_d);
            if(loss_type == 0) {
               vec<GRBLinExpr> exprs;
               for(const auto& yj : y) {
                  exprs.emplace_back(loss_d - loss_ub + ty_d * cls_weight * (yj - yi));
               }
               vec<GRBVar> binaries;
               for(const auto& bi : views::iota(0, int(exprs.size()))) {
                  binaries.emplace_back(model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi)));
               }               
               if(use_bounds) {
                  vec<double> max_bounds;
                  for(const auto& bounds_yj : bounds_y) {
                     max_bounds.emplace_back(-loss_ub + ty_d * cls_weight * (bounds_yj->first - bounds_yi->second));
                  }
                  for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                     set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
                  }
               }
               else {
                  for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                     model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
                  }
               }
               set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
            }
            else if (loss_type == 1) {
               vec<GRBLinExpr> exprs;
               for(const auto& yj : y) {
                  exprs.emplace_back(loss_d - loss_ub + ty_d * (log(max<double>(ty_d, eps)) + yj - yi));
               }
               vec<GRBVar> binaries;
               for(const auto& bi : views::iota(0, int(exprs.size()))) {
                  binaries.emplace_back(model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi)));
               }
               if(use_bounds) {
                  vec<double> max_bounds;
                  for(const auto& bounds_yj : bounds_y) {
                     max_bounds.emplace_back(-loss_ub + ty_d * (log(max<double>(ty_d, eps)) + bounds_yj->first - bounds_yi->second));
                  }
                  for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                     set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
                  }
               }
               else {
                  for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                     model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
                  }
               }
               set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
            }
         }
      }
   }
   GRBVar obj;
   if(is_constr) {
      double obj_ub = use_bounds ? loss_sum_ub - loss_sum_lb : GRB_INFINITY;
      obj = model.addVar(0, obj_ub, 0, GRB_CONTINUOUS, name);
      if(minmax) {
         set_constr(model, obj >= gen_sum_expr(minmax_exprs) - minmax_sum_ub * (1 - *is_constr), minmax_name, lazy);
         obj_ub = max<double>(obj_ub, minmax_sum_ub);
         set_ub(obj, obj_ub);
      }
      if(use_bounds) {
         if(restrict) {
            double max_bound = -size_d * ub_softmax + loss_sum_lb + obj_ub;
            set_constr(model, obj <= size_d * (ub_softmax + eps) - loss_sum_lb + max_bound * (*is_constr), restrict_name, lazy);
         }
         double max_bound = -loss_sum_ub + loss_sum_lb;
         set_constr(model, obj >= gen_sum_expr(exprs) - loss_sum_lb + max_bound * (*is_constr), name, lazy);
      }
      else {
         if(restrict) {
            model.addGenConstrIndicator(*is_constr, 0, obj <= size_d * (ub_softmax + eps) - loss_sum_lb, restrict_name);
         }
         model.addGenConstrIndicator(*is_constr, 0, obj >= gen_sum_expr(exprs) - loss_sum_lb, name);
      }
   }
   else {
      double obj_ub = use_bounds ? loss_sum_ub - loss_sum_lb : GRB_INFINITY;
      if(restrict) {
         obj_ub = min<double>(obj_ub, size_d * (ub_softmax + eps) - loss_sum_lb);
      }
      obj = model.addVar(0, obj_ub, 0, GRB_CONTINUOUS, name);
      set_constr(model, obj >= gen_sum_expr(exprs) - loss_sum_lb, name, lazy);
   }
   return obj;
}

auto gen_class_error_expr(
   GRBModel& model,     const GRBLinExprRange auto& y,       const RangeOf<optional<pair<double, double>>> auto& bounds_y,
   const string& name,  const RangeOf<double> auto& ty,      const RangeOf<double> auto& cls_w, 
   const string& loss,  double eps = 0.0001,                 double rtol = 0.0,
   int priority = 0,    bool restrict = false,               bool minmax = false,
   int lazy = 0,        const optional<double>& recall = {}, const optional<GRBVar>& is_constr = {},
   const optional<double>& slope = {}
) {
   GRBLinExpr expr;
   int size = ranges::distance(y);
   if(size <= 1) {
      expr = gen_sigmoid_error_var(
         model, *ranges::begin(y), name, loss, *ranges::begin(ty), *ranges::begin(cls_w), eps, rtol,
         priority, restrict, minmax, lazy, is_constr, recall, *ranges::begin(bounds_y), slope
      );
   }
   else {
      expr = gen_softmax_error_var(model, y, bounds_y, name, loss, ty, cls_w, eps, rtol, priority, restrict, minmax, lazy, is_constr, slope);
   }
   return expr;
}

auto gen_regression_error_var(
   GRBModel& model,    const GRBLinExpr& y, const string& name,  double ty,
   const string& loss, double eps = 0.0001, double slope = 1.0,  double rtol = 0.0,
   int priority = 0,   int lazy = 0,        bool minmax = false, const optional<GRBVar>& is_constr = {},
   const optional<pair<double, double>>& bounds_y = {}
) {
   vec<GRBLinExpr> exprs, minmax_exprs;
   const auto& minmax_name = format("{}_minmax",   name);
   const auto& loss_name   = format("{}_loss",     name);
   const auto& ic_name     = format("{}_isconstr", name);
   GRBVar loss_i = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, loss_name);
   GRBVar loss_d;
   GRBLinExpr yi = y;
   vec<GRBLinExpr> loss_exprs;
   double loss_i_ub = 0, loss_sum_ub = 0, minmax_sum_ub = 0;
   pair<double, double> bounds_yi;
   if(bounds_y) {
      bounds_yi = *bounds_y;
      if(loss == "Square" || loss == "MSE") {
         loss_i_ub = max<double>(slope * (bounds_yi.first - ty), -slope * (bounds_yi.second - ty));
      }
      else if (loss == "Absolute" || loss == "L1") {
         loss_i_ub = max<double>((bounds_yi.first - ty), -(-bounds_yi.second - ty));
      }
      else if (loss == "Huber") {
         loss_i_ub = max<double>(
            max<double>((0.5 * slope * bounds_yi.first - ty), -0.5 * slope * (bounds_yi.second - ty)),
            max<double>(slope * ( (bounds_yi.first - ty) - 0.5 * slope), slope * (-(bounds_yi.second - ty) - 0.5 * slope))
         );
      }
      else if (loss == "SmoothL1") {
         loss_i_ub = max<double>(
            max<double>((0.5 * bounds_yi.first - ty), -0.5 * (bounds_yi.second - ty)),
            max<double>((bounds_yi.first - ty) - 0.5 * slope, -(bounds_yi.second - ty) - 0.5 * slope)
         );
      }
      loss_sum_ub += loss_i_ub;
   }
   if(loss == "Square" || loss == "MSE") {
      loss_exprs.emplace_back(slope + (yi - ty));
      loss_exprs.emplace_back(slope - (yi - ty));
      loss_exprs.emplace_back(loss_i - slope * (yi - ty));
      loss_exprs.emplace_back(loss_i + slope * (yi - ty));
   }
   else if (loss == "Absolute" || loss == "L1") {
      loss_exprs.emplace_back(loss_i - (yi - ty));
      loss_exprs.emplace_back(loss_i + (yi - ty));
   }
   else if (loss == "Huber") {
      loss_exprs.emplace_back(loss_i - 0.5 * slope *   (yi - ty));
      loss_exprs.emplace_back(loss_i + 0.5 * slope *   (yi - ty));
      loss_exprs.emplace_back(loss_i -       slope * ( (yi - ty) - 0.5 * slope));
      loss_exprs.emplace_back(loss_i -       slope * (-(yi - ty) - 0.5 * slope));
   }
   else if (loss == "SmoothL1") {
      loss_exprs.emplace_back(loss_i - 0.5 * (yi - ty));
      loss_exprs.emplace_back(loss_i + 0.5 * (yi - ty));
      loss_exprs.emplace_back(loss_i -       (yi - ty) - 0.5 * slope);
      loss_exprs.emplace_back(loss_i +       (yi - ty) - 0.5 * slope);
   }
   for(const auto& [li, loss_expr] : loss_exprs | views::enumerate) {
      set_constr(model, loss_expr >= 0, format("{}_{}", loss_name, li), lazy);
   }
   if(is_constr) {
      double ty0 = max<double>(abs(ty), eps);
      double lb = (1 - rtol) * ty0;
      double ub = (1 + rtol) * ty0;
      double loss_ub = GRB_INFINITY;
      for(const auto& ub : {lb, ub}) {
         if(loss == "Square" || loss == "MSE") {
            loss_ub = min<double>(loss_ub, mse_loss(ub, ty));
         }
         else if (loss == "Absolute" || loss == "L1") {
            loss_ub = min<double>(loss_ub, l1_loss(ub, ty));
         }
         else if (loss == "Huber") { 
            loss_ub = min<double>(loss_ub, huber_loss(ub, ty, slope));
         }
         else if (loss == "SmoothL1") {
            loss_ub = min<double>(loss_ub, smooth_l1_loss(ub, ty, slope));
         }
      }
      if(bounds_y) {
         double max_bounds = loss_i_ub - loss_ub;
         set_constr(model, loss_i <= loss_ub + max_bounds * (1 - *is_constr), ic_name, lazy);
      }
      else {
         model.addGenConstrIndicator(*is_constr, 1, loss_i <= loss_ub, ic_name);
      }
      if(minmax) {
         minmax_sum_ub += loss_ub;
         loss_d = model.addVar(0, loss_ub, 0, GRB_CONTINUOUS, minmax_name);
         if(loss == "Square" || loss == "MSE") {
            vec<GRBLinExpr> exprs;
            exprs.emplace_back(loss_d - loss_ub);
            exprs.emplace_back(loss_d - loss_ub + slope * (2 *  (yi - ty) - slope));
            exprs.emplace_back(loss_d - loss_ub + slope * (2 * -(yi - ty) - slope));
            vec<GRBVar> binaries;
            for(const auto& bi : views::iota(0, int(exprs.size()))) {
               binaries.emplace_back(model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi)));
            }
            if(bounds_y) {
               vec<double> max_bounds;
               max_bounds.emplace_back(-loss_ub);
               max_bounds.emplace_back(-loss_ub + slope * (2 *  (bounds_yi.first  - ty) - slope));
               max_bounds.emplace_back(-loss_ub + slope * (2 * -(bounds_yi.second - ty) - slope));
               for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                  set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
               }
            }
            else {
               for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                  model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
               }
            }
            set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
         }
         else if (loss == "Absolute" || loss == "L1") {
            vec<GRBLinExpr> exprs;
            exprs.emplace_back(loss_d - loss_ub);
            exprs.emplace_back(loss_d - loss_ub + (yi - ty));
            exprs.emplace_back(loss_d - loss_ub - (yi - ty));
            vec<GRBVar> binaries;
            for(const auto& bi : views::iota(0, int(exprs.size()))) {
               binaries.emplace_back(model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi)));
            }
            if(bounds_y) {
               vec<double> max_bounds;
               max_bounds.emplace_back(-loss_ub);
               max_bounds.emplace_back(-loss_ub + (bounds_yi.first  - ty));
               max_bounds.emplace_back(-loss_ub - (bounds_yi.second - ty));
               for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                  set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
               }
            }
            else {
               for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                  model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
               }
            }
            set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
         }
         else if (loss == "Huber") {
            vec<GRBLinExpr> exprs;
            exprs.emplace_back(loss_d - loss_ub);
            exprs.emplace_back(loss_d - loss_ub + slope * ( (yi - ty) - 0.5 * slope));
            exprs.emplace_back(loss_d - loss_ub + slope * (-(yi - ty) - 0.5 * slope));
            vec<GRBVar> binaries;
            for(const auto& bi : views::iota(0, int(exprs.size()))) {
               binaries.emplace_back(model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi)));
            }
            if(bounds_y) {
               vec<double> max_bounds;
               max_bounds.emplace_back(-loss_ub);
               max_bounds.emplace_back(-loss_ub + slope * ( (bounds_yi.first  - ty) - 0.5 * slope));
               max_bounds.emplace_back(-loss_ub + slope * (-(bounds_yi.second - ty) - 0.5 * slope));
               for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                  set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
               }
            }
            else {
               for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                  model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
               }
            }
            set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
         }
         else if (loss == "SmoothL1") {
            vec<GRBLinExpr> exprs;
            exprs.emplace_back(loss_d - loss_ub);
            exprs.emplace_back(loss_d - loss_ub + (yi - ty) - 0.5 * slope);
            exprs.emplace_back(loss_d - loss_ub - (yi - ty) - 0.5 * slope);
            vec<GRBVar> binaries;
            for(const auto& bi : views::iota(0, int(exprs.size()))) {
               binaries.emplace_back(model.addVar(0, 1, 0, GRB_BINARY, format("{}_{}", minmax_name, bi)));
            }
            if(bounds_y) {
               vec<double> max_bounds;
               max_bounds.emplace_back(-loss_ub);
               max_bounds.emplace_back(-loss_ub + (bounds_yi.first  - ty) - 0.5 * slope);
               max_bounds.emplace_back(-loss_ub - (bounds_yi.second - ty) - 0.5 * slope);
               for(const auto& [bi, expr, bin, bound] : views::zip(views::iota(0), exprs, binaries, max_bounds)) {
                  set_constr(model, expr >= bound * (1 - bin), format("{}_{}", minmax_name, bi), lazy);
               }
            }
            else {
               for(const auto& [bi, expr, bin] : views::zip(views::iota(0), exprs, binaries)) {
                  model.addGenConstrIndicator(bin, 1, expr >= 0, format("{}_{}", minmax_name, bi));
               }
            }
            set_constr(model, gen_sum_expr(binaries) == 1, format("{}_binaries", minmax_name), lazy);
         }
      }
   }
   GRBVar obj;
   if(is_constr) {
      double obj_ub = 0;
      obj = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
      double max_bounds = -loss_sum_ub;
      set_constr(model, obj >= gen_mean_expr(exprs) + max_bounds * (*is_constr), name, lazy);
      obj_ub = max<double>(obj_ub, loss_sum_ub);
      if(minmax) {
         double max_bounds = -minmax_sum_ub;
         set_constr(model, obj >= gen_mean_expr(minmax_exprs) + max_bounds * (1 - *is_constr), minmax_name, lazy);
         obj_ub = max<double>(obj_ub, minmax_sum_ub);
      }
      set_ub(obj, obj_ub);
   }
   else {
      obj = model.addVar(0, loss_sum_ub, 0, GRB_CONTINUOUS, name);
      set_constr(model, obj >= loss_i, name, lazy);
   }
   return obj;
}

auto gen_regression_error_expr(
   GRBModel& model,        const GRBLinExprRange auto& y,  const RangeOf<optional<pair<double, double>>> auto& bounds_y,
   const string& name,     const RangeOf<double> auto& ty, const string& loss,
   double eps = 0.0001,    double slope = 1.0,             double rtol = 0.0,
   int priority = 0,       int lazy = 0,                   bool minmax = false,
   const optional<GRBVar>& is_constr = {}
) {
   vec<GRBLinExpr> exprs;
   for(const auto& [i, y, ty, bounds_y] : views::zip(views::iota(0), y, ty, bounds_y)) {
      exprs.emplace_back(gen_regression_error_var(model, y, format("{}_{}", name, i), ty, loss, eps, slope, rtol, priority, lazy, minmax, is_constr, bounds_y));
   }
   return gen_mean_expr(exprs);
}

auto gen_bin_w_var(
   GRBModel& model, const GRBVar& b, const GRBLinExpr& a,  const string& name,
   double coef,     int lazy = 0,    bool use_sos = false, const optional<pair<double, double>>& bounds_a = {}
) {
   GRBVar bw;
   const auto& on_name  = format("{}_on",  name);
   const auto& off_name = format("{}_off", name);
   if(bounds_a) {
      const auto& bounds_ac = mult_bounds(*bounds_a, coef);
      const auto& bounds = or_bounds(bounds_ac, 0);
      const auto& bounds_da = sub_bounds(bounds, bounds_ac);
      bw = model.addVar(bounds.first, bounds.second, 0, GRB_CONTINUOUS, name);
      set_constr(model, bw <= coef * a + bounds_da.second *      b,  format("{}_ub", off_name), lazy);
      set_constr(model, bw >= coef * a + bounds_da.first  *      b,  format("{}_lb", off_name), lazy);
      set_constr(model, bw <=            bounds.second    * (1 - b), format("{}_lb", on_name),  lazy);
      set_constr(model, bw >=            bounds.first     * (1 - b), format("{}_ub", on_name),  lazy);
   }
   else {
      bw = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
      model.addGenConstrIndicator(b, 0, bw == coef * a, off_name);
      if(use_sos) {
         array<GRBVar, 2> vars_b{b, bw}; array<double, 2> weights_b{0, 1};
         model.addSOS(vars_b.data(), weights_b.data(), 2, GRB_SOS_TYPE1);
      }
      else {
         model.addGenConstrIndicator(b, 1, bw == 0, on_name);
      }
   }
   return bw;
}

GRBLinExpr gen_bin_w_expr(
   GRBModel& model, const GRBVar& b, const GRBLinExpr& a,  const string& name,
   double coef,     int lazy = 0,    bool use_sos = false, const optional<pair<double, double>>& bounds_a = {}
) {
   if(coef == 0.0) {
      return GRBLinExpr();
   }
   return gen_bin_w_var(model, b, a, name, coef, lazy, use_sos, bounds_a);
}

GRBLinExpr gen_act_w_expr(const GRBLinExprRange auto& bw) {
   return accumulate(ranges::begin(bw) + 1, ranges::end(bw), -*ranges::begin(bw));
}

GRBVar gen_w_semi_var(GRBModel& model, const RangeOf<GRBVar> auto& b, const string& name, int priority = 0, int lazy = 0) {
   int bits = ranges::distance(b);
   GRBVar w_i = model.addVar(-exp2(bits - 1), exp2(bits - 1) - 1, 0, GRB_INTEGER, format("{}_wi", name));
   vec<GRBLinExpr> exp_exprs;
   for(const auto& [l, b] : b | views::enumerate) {
      exp_exprs.emplace_back(exp2(bits - 1 - l) * (1 - b));
   }
   set_constr(model, w_i == accumulate(exp_exprs.begin() + 1, exp_exprs.end(), -exp_exprs[0]), format("{}_wi", name), lazy);
   return w_i;
}

GRBLinExpr gen_w_expr(
   GRBModel& model, const GRBVar& wi, const GRBVar& iw,
   int bits = 4,    int exp = 4,      const optional<double>& tw = {}
) {
   GRBLinExpr expr = exp2(exp - bits + 1) * wi;
   if(tw) {
      expr += *tw * (1 - iw);
   }
   return expr;
}

GRBLinExpr gen_l1w_expr(
   GRBModel& model,    const MatrixRange<GRBLinExpr> auto& w, const MatrixRange<pair<double, double>> auto& bounds_w,
   const string& name, int lazy = 0,    bool use_grb = false, bool use_bounds = false                  
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
      for(const auto& [j, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
         GRBVar loss_d;
         if(use_bounds) {
            const auto& bounds_abs = abs_bounds(bounds_w);
            loss_d = model.addVar(bounds_abs.first, bounds_abs.second, 0, GRB_CONTINUOUS, format("{}_{}_{}", name, i, j));
         }
         else {
            loss_d = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_{}_{}", name, i, j));
         }
         vec<GRBLinExpr> loss_exprs;
         loss_exprs.emplace_back(loss_d + w);
         loss_exprs.emplace_back(loss_d - w);
         for(const auto& [k, expr] : loss_exprs | views::enumerate) {
            set_constr(model, expr >= 0, format("{}_{}_{}_{}", name, i, j, k), lazy);
         }
         exprs.emplace_back(loss_d);
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l2w_expr(
   GRBModel& model,    const MatrixRange<GRBLinExpr> auto& w, const MatrixRange<pair<double, double>> auto& bounds_w,
   const string& name, int lazy = 0,    bool use_grb = false, bool use_bounds = false
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
      for(const auto& [j, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
         GRBVar loss_d;
         if(use_bounds) {
            const auto& bounds_sqr = max_bounds(add_bounds(mult_bounds(abs_bounds(bounds_w), 2), -1), 0);
            loss_d = model.addVar(bounds_sqr.first, bounds_sqr.second, 0, GRB_CONTINUOUS, format("{}_{}_{}", name, i, j));
         }
         else {
            loss_d = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_{}_{}", name, i, j));
         }
         vec<GRBLinExpr> loss_exprs;
         loss_exprs.emplace_back(loss_d - ( 2 * w - 1));
         loss_exprs.emplace_back(loss_d - (-2 * w - 1));
         for(const auto& [k, expr] : loss_exprs | views::enumerate) {
            set_constr(model, expr >= 0, format("{}_{}_{}_{}", name, i, j, k), lazy);
         }
         exprs.emplace_back(loss_d);
      }
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l1a_expr(
   GRBModel& model,    const RangeOf<GRBLinExpr> auto& a,  const RangeOf<optional<pair<double, double>>> auto& bounds_a,
   const string& name, int lazy = 0, bool use_grb = false
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, a, bounds_a] : views::zip(views::iota(0), a, bounds_a)) {
      GRBVar loss_d;
      if(bounds_a) {
         const auto& bounds_abs = abs_bounds(*bounds_a);
         loss_d = model.addVar(bounds_abs.first, bounds_abs.second, 0, GRB_CONTINUOUS, format("{}_{}", name, i));
      }
      else {
         loss_d = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_{}", name, i));
      }
      vec<GRBLinExpr> loss_exprs;
      loss_exprs.emplace_back(loss_d + a);
      loss_exprs.emplace_back(loss_d - a);
      for(const auto& [k, expr] : loss_exprs | views::enumerate) {
         set_constr(model, expr >= 0, format("{}_{}_{}", name, i, k), lazy);
      }
      exprs.emplace_back(loss_d);
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_l2a_expr(
   GRBModel& model,    const RangeOf<GRBLinExpr> auto& a, const RangeOf<optional<pair<double, double>>> auto& bounds_a,
   const string& name, int lazy = 0,                      bool use_grb = false
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, a, bounds_a] : views::zip(views::iota(0), a, bounds_a)) {
      GRBVar loss_d;
      if(bounds_a) {
         const auto& bounds_sqr = max_bounds(add_bounds(mult_bounds(abs_bounds(*bounds_a), 2), -1), 0);
         loss_d = model.addVar(bounds_sqr.first, bounds_sqr.second, 0, GRB_CONTINUOUS, format("{}_{}", name, i));
      }
      else {
         loss_d = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_{}", name, i));
      }
      vec<GRBLinExpr> loss_exprs;
      loss_exprs.emplace_back(loss_d - ( 2 * a - 1));
      loss_exprs.emplace_back(loss_d - (-2 * a - 1));
      for(const auto& [k, expr] : loss_exprs | views::enumerate) {
         set_constr(model, expr >= 0, format("{}_{}_{}", name, i, k), lazy);
      }
      exprs.emplace_back(loss_d);
   }
   return gen_mean_expr(exprs);
}

GRBLinExpr gen_wp_expr(
   GRBModel& model,    const MatrixRange<GRBLinExpr> auto& w,        const MatrixRange<pair<double, double>> auto& bounds_w,
   const string& name, const MatrixRange<optional<double>> auto& tw, bool use_sqr = false,
   int lazy = 0,       bool use_grb = false,                         bool use_bounds = false
) {
   vec<GRBLinExpr> exprs;
   for(   const auto& [i, w, tw, bounds_w] : views::zip(views::iota(0), w, tw, bounds_w)) {
      for(const auto& [j, w, tw, bounds_w] : views::zip(views::iota(0), w, tw, bounds_w)) {
         if(tw) {
            GRBVar loss_d;
            if(use_bounds) {
               pair<double, double> bounds_loss;
               if(!use_sqr) {
                  bounds_loss = abs_bounds(abs_bounds(bounds_w));
               }
               else {
                  bounds_loss = max_bounds(add_bounds(mult_bounds(abs_bounds(bounds_w), 2), -1), 0);
               }
               loss_d = model.addVar(bounds_loss.first, bounds_loss.second, 0, GRB_CONTINUOUS, format("{}_{}_{}", name, i, j));
            }
            else {
               loss_d = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_{}_{}", name, i, j));
            }
            vec<GRBLinExpr> loss_exprs;
            if(!use_sqr) {
               loss_exprs.emplace_back(loss_d + w);
               loss_exprs.emplace_back(loss_d - w);
            }
            else {
               loss_exprs.emplace_back(loss_d - ( 2 * w - 1));
               loss_exprs.emplace_back(loss_d - (-2 * w - 1));
            }
            for(const auto& [k, expr] : loss_exprs | views::enumerate) {
               set_constr(model, expr >= 0, format("{}_{}_{}_{}", name, i, j, k), lazy);
            }
            exprs.emplace_back(loss_d);
         }
      }
   }
   return gen_mean_expr(exprs);
}

auto gen_class_count(const Tensor3Range<double> auto& ty, double eps = 2.09e-9) {
   double instances = ranges::distance(ty);
   mat<double> cls_w(ranges::distance(*ranges::begin(ty)), vec<double>(ranges::distance(*ranges::begin(*ranges::begin(ty))), 0.0));
   for(const auto& ty : ty) {
      for(auto&& [ty, cls_w] : views::zip(ty, cls_w)) {
         for(auto&& [ty, cls_w] : views::zip(ty, cls_w)) {
            cls_w += ty;
         }
      }
   }
   for(auto& cls_w: cls_w) {
      for(auto& cls_w : cls_w) {
         cls_w = instances / max<double>(cls_w, eps);
      }
   }
   return cls_w;
}

auto linear_bounds(const RangeOf<pair<double, double>> auto& a, const MatrixRange<pair<double, double>> auto& w, double bias = 1) {
   int j_max = -1;    mat<pair<double, double>> aw;
   vec<pair<double, double>> input;
   ranges::copy(a, back_inserter(input));
   input.emplace_back(make_pair(bias, bias));
   for(   const auto& [i, w, a] : views::zip(views::iota(0), w, input)) {
      for(const auto& [j, w]    : views::zip(views::iota(0), w)) {
         if(j_max < j) {
            aw.emplace_back(vec<pair<double, double>>());
            j_max = j;
         }
         aw[j].emplace_back(mult_bounds(w, a));
      }
   }
   vec<pair<double, double>> z;
   for(const auto& aw : aw) {
      z.emplace_back(make_pair(0.0, 0.0));
      for(const auto& aw : aw) {
         z.back() = add_bounds(aw, z.back());
      }
   }
   return z;
}

auto linear_layer(
   GRBModel& model,    const RangeOf<double> auto& a, const MatrixRange<GRBLinExpr> auto& w,
   const string& name, double bias = 1
) {
   int j_max = -1;    mat<GRBLinExpr> aw;
   vec<double> input;
   ranges::copy(a, back_inserter(input));
   input.emplace_back(bias);
   for(   const auto& [i, w, a] : views::zip(views::iota(0), w, input)) {
      for(const auto& [j, w]    : views::zip(views::iota(0), w)) {
         if(j_max < j) {
            aw.emplace_back(vec<GRBLinExpr>());
            j_max = j;
         }
         aw[j].emplace_back(w * a);
      }
   }
   vec<GRBLinExpr> z;
   ranges::move(aw | views::transform(gen_sum_expr<decltype(aw)::value_type>), back_inserter(z));
   return z;
}

auto linear_layer(
   GRBModel& model,                    const RangeOf<GRBLinExpr> auto& a,   const RangeOf<optional<pair<double, double>>> auto& bounds_a,
   const Tensor3Range<GRBVar> auto& b, const MatrixRange<GRBVar> auto& iw,  const MatrixRange<GRBLinExpr> auto& w, 
   const MatrixRange<int> auto& exp,   const MatrixRange<bool> auto& fixed, const MatrixRange<optional<double>> auto& tw,
   const string& name,                 int lazy = 0,                        bool local = false, 
   bool use_sos = false,               double bias = 1
) {
   mat<GRBLinExpr> aw;   int j_max = -1;
   for(   const auto& [i, b, w, iw, tw, exp, fixed, bounds_a, a] : views::zip(views::iota(0), b, w, iw, tw, exp, fixed, bounds_a, a)) {
      for(const auto& [j, b,    iw, tw, exp, fixed]    : views::zip(views::iota(0), b,    iw, tw, exp, fixed)) {
         if(j_max < j) {
            aw.emplace_back(vec<GRBLinExpr>());
            aw[j].emplace_back(bias * w.back());
            j_max = j;
         }
         GRBLinExpr actw;
         if(tw) {
            if(fixed || local) {
               actw += *tw * a;
            }
            else {
               actw += gen_bin_w_expr(
                  model, iw, a, format("{}_iwa_{}_{}", name, i, j),
                  *tw, lazy, use_sos, bounds_a
               );
            }
         }
         if(!fixed) {
            vec<GRBLinExpr> bw;
            for(const auto& [l, b] : b | views::enumerate) {
               bw.emplace_back(gen_bin_w_expr(
                  model, b, a, format("{}_bw_{}_{}_{}", name, i, j, l),
                  exp2(exp - l), lazy, use_sos, bounds_a
               ));
            }
            actw += gen_act_w_expr(bw);
         }
         aw[j].emplace_back(actw);
      }
   }
   vec<GRBLinExpr> z;
   ranges::move(aw | views::transform(gen_sum_expr<decltype(aw)::value_type>), back_inserter(z));
   return z;
}

auto gen_layers(
   GRBModel& model, int layers,               const RangeOf<int> auto& cap,
   const Tensor3Range<bool> auto& mask,       const Tensor3Range<int> auto& exp,
   const Tensor3Range<int> auto& bits,        const Tensor3Range<optional<double>> auto& tw,
   const RangeOf<optional<double>> auto& l1w, const RangeOf<optional<double>> auto& l2w,
   const optional<double>& wpen,              const Tensor3Range<bool> auto& fixed,
   double eps = 2.09e-9,                      bool local = false,
   bool no_l1 = false,                        bool no_l2 = false,
   bool use_sqr = false,                      bool use_grb = false,
   const optional<int>& lazy = {},            bool use_bounds = false,
   const optional<pair<double, double>>& sparsity = {},
   bool use_asym = false
) {
   int bin_priority = 5, weight_priority = 6, sparsity_priority = 7, asym_priority = 8;
   GRBVar one_var  = model.addVar(1, 1, 0, GRB_BINARY, "one_singleton" );
   GRBVar zero_var = model.addVar(0, 0, 0, GRB_BINARY, "zero_singleton");
   ten4<GRBVar> b(layers); ten3<GRBVar> iw(layers);
   ten3<GRBLinExpr> w(layers);
   vec<GRBLinExpr> mask_exprs, l1w_exprs, l2w_exprs, wpen_exprs;
   ten3<pair<double, double>> bounds_w(layers);
   for(    auto&& [k, b, tw, bits, mask, exp, fixed, l1w, l2w, iw, w, bounds_w, sizes] : views::zip(
      views::iota(0), b, tw, bits, mask, exp, fixed, l1w, l2w, iw, w, bounds_w, cap | views::adjacent<2>
   )) {
      resetline_console();
      cout << "Processing layer " << k << "..." << flush;
      const auto& [n, m] = sizes;
      b  = ten3<GRBVar>(n + 1, mat<GRBVar>(m));
      iw = mat<GRBVar>(n + 1, vec<GRBVar>(m, one_var));
      w  = mat<GRBLinExpr>(n + 1, vec<GRBLinExpr>(m));
      bounds_w = mat<pair<double, double>>(n + 1, vec<pair<double, double>>(m));
      mat<GRBVar> bt(m), iwt(m), wt(m);
      for(   auto&& [i, b, w, tw, bits, exp, mask, fixed, iw, bounds_w] : views::zip(views::iota(0), b, w, tw, bits, exp, mask, fixed, iw, bounds_w)) {
         for(auto&& [j, b, w, tw, bits, exp, mask, fixed, iw, bounds_w] : views::zip(views::iota(0), b, w, tw, bits, exp, mask, fixed, iw, bounds_w)) {
            b = vec<GRBVar>(bits + 1, one_var);
            if(local || fixed) {
               iw = zero_var;
            }
            if(!fixed && mask) {
               if(tw && !local) {
                  iw = model.addVar(0, 1, 0, GRB_BINARY, format("iw_{}_{}_{}", k, i, j));
                  set_priority(iw, weight_priority * layers + k);
               }
               for(auto&& [l, b] : b | views::enumerate) {
                  b = model.addVar(0, 1, 0, GRB_BINARY, format("b_{}_{}_{}_{}", k, i, j, l));
                  set_priority(b, bin_priority * layers + k);
               }
            }
            GRBVar ws = gen_w_semi_var(model, b, format("ws_{}_{}_{}", k, i, j), weight_priority, lazy.value_or(0));
            w = gen_w_expr(model, ws, iw, bits + 1, exp, tw);
            int b0 = ranges::distance(b);
            if(use_bounds) {
               bounds_w.first  = -exp2(exp);
               bounds_w.second =  exp2(exp) - exp2(exp - b0);
               if(tw) {
                  bounds_w = add_bounds(bounds_w, or_bounds(make_pair(*tw, *tw), 0));
               }
            }
         }
      }
   }
   if(sparsity) {
      ten3<GRBVar> w0;
      for(const auto& [k, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
         w0.emplace_back(mat<GRBVar>(ranges::distance(w)));
         for(const auto& [i, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
            w0.back().emplace_back(vec<GRBVar>(ranges::distance(w)));
            for(const auto& [j, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
               GRBVar b = model.addVar(0, 1, 0, GRB_BINARY, format("w0_{}_{}_{}", k, i, j));
               set_priority(b, sparsity_priority * layers + k);
               set_constr(model,  w <=  bounds_w.second * (1 - b), format("w0ub_{}_{}_{}", k, i, j), lazy.value_or(0));
               set_constr(model, -w <= -bounds_w.first  * (1 - b), format("w0lb_{}_{}_{}", k, i, j), lazy.value_or(0));
               w0.back().back().emplace_back(b);
            }
         }
         const auto& joined_w = w0.back() | views::join;
         gen_range_constr(
            model, gen_sum_expr(joined_w),
            floor(ranges::distance(joined_w) * sparsity->first),
            floor(ranges::distance(joined_w) * sparsity->second),
            format("sparsity_{}", k), lazy.value_or(1), use_grb
         );
      }
   }
   if(use_asym) {
      for(const auto& [k, w, bounds_w, exp, bits, tw, m] : views::zip(views::iota(0), w, bounds_w, exp, bits, tw, cap | views::drop(1))) {
         for(const auto& [ja, jb] : views::iota(0, m) | views::adjacent<2>) {
            vec<GRBVar> binaries;
            for(const auto& [i, w, bounds_w, exp, bits, tw] : views::zip(views::iota(0), w, bounds_w, exp, bits, tw)) {
               GRBVar b = model.addVar(0, 1, 0, GRB_BINARY, format("as_{}_{}_{}_{}", k, i, ja, jb));
               set_priority(b, asym_priority * layers + k);
               double w_eps = exp2(min<double>(exp[ja] - (bits[ja] + 1), exp[jb] - (bits[jb] + 1)));
               if(tw[ja]) {
                  w_eps = min<double>(w_eps, abs(*tw[ja]));
               }
               if(tw[jb]) {
                  w_eps = min<double>(w_eps, abs(*tw[jb]));
               }
               set_constr(model, w[ja] - w[jb] >= w_eps + (bounds_w[ja].first - bounds_w[jb].second - w_eps) * (1 - b), format("as_{}_{}_{}", k, i, ja), lazy.value_or(0));
               binaries.emplace_back(b);
            }
            for(const auto& [i, w, bounds_w] : views::zip(views::iota(0), w, bounds_w)) {
               set_constr(model, w[ja] - w[jb] >= (bounds_w[ja].first  - bounds_w[jb].second) * (1 - gen_sum_expr(binaries | views::drop(i + 1))),  format("aslb_{}_{}_{}", k, i, ja), lazy.value_or(0));
               set_constr(model, w[ja] - w[jb] <= (bounds_w[ja].second - bounds_w[jb].first ) * (1 - gen_sum_expr(binaries | views::drop(i + 1))),  format("asub_{}_{}_{}", k, i, ja), lazy.value_or(0));
            }
            set_constr(model, gen_sum_expr(binaries) == 1, format("as_{}_{}", k, ja), lazy.value_or(0));
         }
      }
   }
   if(wpen) {
      for(const auto& [k, w, tw, bounds_w] : views::zip(views::iota(0), w, tw, bounds_w)) {
         wpen_exprs.emplace_back(*wpen * gen_wp_expr(
            model, w, bounds_w, format("wp_{}", k), tw, 
            use_sqr, lazy.value_or(0), use_grb, use_bounds
         ));
      }
   }
   if(!no_l1) {
      for(const auto& [k, w, l1w, bounds_w] : views::zip(views::iota(0), w, l1w, bounds_w) | views::take(layers - 1)) {
         if(l1w) {
            l1w_exprs.emplace_back(*l1w * gen_l1w_expr(
               model, w, bounds_w, format("l1w_{}", k),
               lazy.value_or(0), use_grb, use_bounds
            ));
         }
      }
   }
   if(!no_l2) {
      for(const auto& [k, w, l2w, bounds_w] : views::zip(views::iota(0), w, l2w, bounds_w) | views::take(layers - 1)) {
         if(l2w) {
            l2w_exprs.emplace_back(*l2w * gen_l2w_expr(
               model, w, bounds_w, format("l2w_{}", k),
               lazy.value_or(0), use_grb, use_bounds
            ));
         }
      }
   }
   return make_tuple(b, l1w_exprs, l2w_exprs, wpen_exprs, iw, w, bounds_w);
}

auto gen_model(
   GRBModel& model,                    int instances, int layers,                     const RangeOf<int> auto& cap,
   const RangeOf<string> auto& af,     const Tensor3Range<bool> auto& mask,           const Tensor3Range<int> auto& exp,
   const Tensor3Range<int> auto& bits, const Tensor3Range<optional<double>> auto& tw, const RangeOf<double> auto& bias,
   const MatrixRange<double> auto& params,                                            const RangeOf<optional<double>> auto& l1a,
   const RangeOf<optional<double>> auto& l1w,                                         const RangeOf<optional<double>> auto& l2a,
   const RangeOf<optional<double>> auto& l2w,                                         const optional<double>& wpen,
   const Tensor3Range<bool> auto& fixed,                                              const MatrixRange<double> auto& fx,
   const Tensor3Range<double> auto& reg_ty,                                           const Tensor3Range<double> auto& class_ty,
   const RangeOf<string> auto& reg_loss,                                              const RangeOf<string> auto& cls_loss,
   double eps = 0.000000000001,        double rtol = 0.1,                             double offtol = 0.0,
   double err_prio = 1.0,              bool max_constrs = false,
   bool min_constrs = false,           bool restrict = false,                         bool minmax = false,
   bool local = false,                 bool no_l1 = false,                            bool no_l2 = false,
   bool use_sqr = false,               bool use_grb = false,                          bool use_sos = false,
   bool use_bounds = false,            bool use_asym = false,
   const optional<double>& bound = {},
   const optional<pair<double, double>>& sparsity = {},                               const optional<pair<double, double>>& relax_frac = {},
   const optional<int>& lazy = {},     const optional<double>& slope = {},            const optional<double>& recall = {},
   const optional<int>& lim_bits = {}, const optional<double>& log_slope = {}
) {
   int activation_priority = 10, zero_priority = 9, relax_priority = 0, cls_priority = 8;
   resetline_console();
   cout << "Processing layers variables..." << flush;
   use_bounds = bound || use_bounds;
   auto&& [b, l1w_exprs, l2w_exprs, wpen_exprs, iw, w, bounds_w] = gen_layers(
      model, layers, cap, mask, exp, bits, tw, l1w, l2w, wpen, fixed, eps,
      local, no_l1, no_l2, use_sqr, use_grb, lazy, use_bounds, sparsity, use_asym
   );
   vec<GRBLinExpr> l1a_exprs, l2a_exprs, cls_target_exprs, reg_target_exprs;
   const auto& cls_w = gen_class_count(class_ty, eps);
   vec<GRBVar> constrs;
   for(const auto& [t, fx, cls, reg] : views::zip(views::iota(0), fx, class_ty, reg_ty)) {
      resetline_console();
      cout << "Processing instance " << t << "...\n" << flush;
      mat<GRBLinExpr> a(layers + 1);
      mat<optional<pair<double, double>>> bounds_a(layers + 1);
      vec<double> fx_i;
      for(const auto& fx : fx) {
         double value = lim_bits ? limit_bits(fx, *lim_bits) : fx;
         a[0].emplace_back(value);
         fx_i.emplace_back(value);
         bounds_a[0].emplace_back(make_pair(value, value));
      }
      for(
         auto&&                 [k, b, w, iw, tw,         atup,           bias, l1a, l2a, exp, af, params, fixed, bounds_w, bounds_tup] :
         views::zip(views::iota(0), b, w, iw, tw, a | views::adjacent<2>, bias, l1a, l2a, exp, af, params, fixed, bounds_w, bounds_a | views::adjacent<2>)
      ) {
         resetline_console();
         cout << "Processing layer " << k << "..." << flush;
         auto&& [bounds_a0, bounds_a1] = bounds_tup;
         auto&& [z0, z1] = atup;
         double bias_d = lim_bits ? limit_bits(bias, *lim_bits) : bias;
         const auto& linear_name = format("linear_{}_{}", t, k);
         vec<GRBLinExpr> z;
         if(k > 0) {
            z = linear_layer(model, z0, bounds_a0, b, iw, w, exp, fixed, tw, linear_name, lazy.value_or(0), local, use_sos, bias_d);
         }
         else {
            z = linear_layer(model, fx_i, w, linear_name, bias_d);
         }
         vec<optional<pair<double, double>>> bounds_z;
         if(use_bounds) {
            if(bound) {
               bounds_z = vec<optional<pair<double, double>>>(z.size(), make_pair(-*bound, *bound));
            }
            else {
               vec<pair<double, double>> bounds_ai;
               for(const auto& bounds : bounds_a0) {
                  bounds_ai.emplace_back(bounds.value_or(make_pair(0.0, 0.0)));
               }
               ranges::copy(linear_bounds(bounds_ai, bounds_w, bias_d), back_inserter(bounds_z));
            }
         }
         else {
            bounds_z = vec<optional<pair<double, double>>>(z.size());
         }
         z1 = gen_activation_exprs(
            model, af, z, bounds_z, format("act_{}_{}", t, k), params, lazy.value_or(0),
            activation_priority * layers + k, use_grb, use_sos
         );
         if(use_bounds) {
            if(bound) {
               bounds_a1 = vec<optional<pair<double, double>>>(ranges::distance(z1), make_pair(-*bound, *bound));
            }
            else {
               bounds_a1 = activation_bounds(af, bounds_z, params);
            }
         }
         else {
            bounds_a1 = vec<optional<pair<double, double>>>(ranges::distance(z1));
         }
      }
      optional<GRBVar> is_constr;
      if(relax_frac) {
         is_constr = model.addVar(0, 1, 0, GRB_BINARY, format("isconstr_{}", t));
         set_priority(*is_constr, relax_priority * layers);
         constrs.emplace_back(*is_constr);
      }
      int asize = 0;
      vec<GRBLinExpr> l1a_exprs_i, l2a_exprs_i, cls_target_exprs_i, reg_target_exprs_i;
      for(const auto& [ti, ty, cls_w, loss] : views::zip(views::iota(0), cls, cls_w, cls_loss)) {
         int size = ranges::distance(ty);
         if(size > 0) {
            vec<double> targets;
            for(const auto& ty : ty) {
               targets.emplace_back(lim_bits ? limit_bits(ty, *lim_bits) : ty);
            }
            cls_target_exprs_i.emplace_back(gen_class_error_expr(
               model, a.back() | views::drop(asize) | views::take(size) | views::common,
               bounds_a.back() | views::drop(asize) | views::take(size) | views::common,
               format("cls_{}_{}", t, ti), targets, cls_w, loss,
               eps, rtol, cls_priority * layers, restrict, minmax, lazy.value_or(0),
               recall, is_constr, log_slope
            ));
            asize += size;
         }
      }
      for(const auto& [ti, ty, loss] : views::zip(views::iota(0), reg, reg_loss)) {
         int size = ranges::distance(ty); 
         if(size > 0) {
            vec<double> targets;
            for(const auto& ty : ty) {
               targets.emplace_back(lim_bits ? limit_bits(ty, *lim_bits) : ty);
            }
            reg_target_exprs_i.emplace_back(gen_regression_error_expr(
               model, a.back() | views::drop(asize) | views::take(size) | views::common,
               bounds_a.back() | views::drop(asize) | views::take(size) | views::common,
               format("reg_{}_{}", t, ti), targets, loss, eps, slope.value_or(2.0), rtol, cls_priority * layers,
               lazy.value_or(0), minmax, is_constr
            ));
            asize += size;
         }
      }
      if(!no_l1) {
         for(const auto& [k, a, bounds_a, l1a] : views::zip(views::iota(0), a | views::drop(1) | views::take(layers - 1), bounds_a | views::drop(1), l1a)) {
            if(l1a) {
               l1a_exprs_i.emplace_back(*l1a * gen_l1a_expr(
                  model, a, bounds_a, format("l1a_{}_{}", t, k), lazy.value_or(0), use_grb
               ));
            }
         }
         l1a_exprs.emplace_back(gen_mean_expr(l1a_exprs_i));
      }
      if(!no_l2) {
         for(const auto& [k, a, bounds_a, l2a] : views::zip(views::iota(0), a | views::drop(1) | views::take(layers - 1), bounds_a | views::drop(1), l2a)) {
            if(l2a) {
               l2a_exprs_i.emplace_back(*l2a * gen_l2a_expr(
                  model, a, bounds_a, format("l2a_{}_{}", t, k), lazy.value_or(0), use_grb
               ));
            }
         }
         l2a_exprs.emplace_back(gen_mean_expr(l2a_exprs_i));
      }
      if(cls_target_exprs_i.size() > 0) {
         cls_target_exprs.emplace_back(gen_mean_expr(cls_target_exprs_i));
      }
      if(reg_target_exprs_i.size() > 0) {
         reg_target_exprs.emplace_back(gen_mean_expr(reg_target_exprs_i));
      }
      resetline_console();     cursorup_console(1);
   }
   GRBLinExpr cls_expr = gen_sum_expr(cls_target_exprs);
   GRBLinExpr reg_expr = gen_sum_expr(reg_target_exprs);
   GRBLinExpr instance_expr = cls_expr + reg_expr;
   GRBLinExpr relax_expr;
   if(relax_frac) {
      double lb = clamp<double>(floor(relax_frac->first  * instances), 1, instances - 1);
      double ub = clamp<double>(floor(relax_frac->second * instances), 1, instances - 1);
      GRBLinExpr relax_constrs_expr = instances - gen_sum_expr(constrs);
      if(ub - lb >= 1) {
         gen_range_constr(model, relax_constrs_expr, lb, ub, "relaxed", lazy.value_or(0), use_grb);
         if(min_constrs || max_constrs) {
            GRBVar best_zero = model.addVar(0, 1, 0, GRB_BINARY, format("bz"));
            set_priority(best_zero, (relax_priority + 1) * layers);
            const auto& on_name = format("bz1");
            if(bound) {
               double max_bound = *bound * instances * cap.back();
               set_constr(model, instance_expr <= max_bound * (1 - best_zero), on_name, lazy.value_or(0));
            }
            else {
               model.addGenConstrIndicator(best_zero, 1, instance_expr <= 0, on_name);
            }
            const auto& on_relax_name  = format("mr1");           const auto& off_relax_name = format("mr0");
            GRBVar relax_var = model.addVar(lb, ub, 0, GRB_INTEGER, format("relaxed_instances"));
            set_priority(relax_var, (relax_priority + 2) * layers);
            set_constr(model, relax_var <= relax_constrs_expr + (ub - lb) * (1 - best_zero), format("{}_ub",  on_relax_name),  lazy.value_or(0));
            set_constr(model, relax_var >= relax_constrs_expr - (ub - lb) * (1 - best_zero), format("{}_lb",  on_relax_name),  lazy.value_or(0));
            set_constr(model, relax_var <=                 lb + (ub - lb) *      best_zero,  format("{}_var", off_relax_name), lazy.value_or(0));
            if(min_constrs) {
               relax_expr += (lb - relax_var) / (ub - lb);
            }
            if(max_constrs) {
               relax_expr += (relax_var - ub) / (ub - lb);
            }
         }
      }
      else if(ub == lb) {
         set_constr(model, ub == relax_constrs_expr, "relaxed",  lazy.value_or(0));
      }
   }
   resetline_console();    cursorup_console(1);      resetline_console();
   model.setObjective(
      instances * layers * (
         gen_mean_expr(wpen_exprs) +
         gen_mean_expr(l1a_exprs) + gen_mean_expr(l1w_exprs) +
         gen_mean_expr(l2a_exprs) + gen_mean_expr(l2w_exprs)
      ) + err_prio * layers * instance_expr + relax_expr,
      GRB_MINIMIZE
   );
   return make_tuple(b, iw);
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
   string suffix7z = "";
   if(opts.contains("7z")) {
      suffix7z = ".7z";
   }
   // Procesa las rutas de los archivos a leer
   bool index       = !opts.contains("no_index" );
   bool header      = !opts.contains("no_header");
   path save_path   = opts.contains("save_path") ? opts["save_path"][0] : ".";
   path load_path   = opts.contains("load_path") ? opts["load_path"][0] : ".";
   string save_name = opts.contains("save_name") ? opts["save_name"][0] : "model";
   string load_name = opts.contains("load_name") ? opts["load_name"][0] : "";
   // Lee las caracteristicas de la arquitectura y la base de datos
   path arch_path     = load_path / format("{}.csv", safe_suffix(load_name, "arch"));
   path features_path = load_path / format("{}.csv", safe_suffix(load_name, "ftr"));
   string def_cls_loss = "LogLoss";
   process_yes_arg(opts, "cls_loss", [&def_cls_loss](const auto& args) {
      stringstream(args[0]) >> def_cls_loss;
   });
   string def_reg_loss = "Absolute";
   process_yes_arg(opts, "reg_loss", [&def_reg_loss](const auto& args) {
      stringstream(args[0]) >> def_reg_loss;
   });
   const auto& [regression_targets, reg_loss] = get_targets(get_targets_paths(load_path, "reg_tgt"), header, index, def_reg_loss);
   const auto& [class_targets,      cls_loss] = get_targets(get_targets_paths(load_path, "cls_tgt"), header, index, def_cls_loss);
   vec<int> capacity;
   vec<string> activation;
   vec<double> bias;
   vec<optional<double>> l1w, l2w, l1a, l2a;
   const auto& features = read_matrix_from_csv<double>(fstream(features_path), header, index);
   if(opts.contains("neurons") && opts.contains("layers") && opts.contains("activation") && opts.contains("bias")) {
      int L, C;
      double B;
      string AF;
      stringstream(opts["layers"][0]) >> L;
      stringstream(opts["neurons"][0]) >> C;
      stringstream(opts["activation"][0]) >> AF;
      stringstream(opts["bias"][0]) >> B; 
      int targets_size = 0;
      for(const auto& ty : class_targets) {
         targets_size += ranges::distance(ty.back());
      }
      for(const auto& ty : regression_targets) {
         targets_size += ranges::distance(ty.back());
      }
      capacity.emplace_back(ranges::distance(features.back()));
      for(const auto& i : views::iota(0, L - 1)) {
         capacity.emplace_back(C);
         activation.emplace_back(AF);
         bias.emplace_back(B);
         l1w.emplace_back(optional<double>());
         l1a.emplace_back(optional<double>());
         l2w.emplace_back(optional<double>());
         l2a.emplace_back(optional<double>());
      }
      capacity.emplace_back(targets_size);
      activation.emplace_back("None");
      activation.emplace_back("None");
      l1w.emplace_back(optional<double>());
      l1w.emplace_back(optional<double>());
      l1a.emplace_back(optional<double>());
      l1a.emplace_back(optional<double>());
      l2w.emplace_back(optional<double>());
      l2w.emplace_back(optional<double>());
      l2a.emplace_back(optional<double>());
      l2a.emplace_back(optional<double>());
      bias.emplace_back(B);
      bias.emplace_back(B);
   }
   else {
      const auto& [C, AF, B, l1w_norm, l1a_norm, l2w_norm, l2a_norm] = read_arch(fstream(arch_path));
      capacity = C;   activation = AF;  bias = B;
      l1w = l1w_norm; l1a = l1a_norm;
      l2w = l2w_norm; l2a = l2a_norm;
   }
   int L = capacity.size() - 1;
   cout << "Numero de capas " << L << "\n";
   for(const auto& [k, sizes] : capacity | views::adjacent<2> | views::enumerate) {
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
   int max_exp = 2;
   process_yes_arg(opts, "max_exp", [&max_exp](const auto& args) {
      stringstream(args[0]) >> max_exp;
   });
   vec<mat<int>> bits;
   if(opts.contains("bits")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "bits"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      bits = clamp_layers_matrix(get_layers_matrix<int, int>(data, capacity), min_bits, max_bits);
   }
   if(bits.empty()) {
      cout << "Trying with default bits\n";
      bits = full_layer_parameter<int>(capacity, max_bits);
   }
   // Lee la precision o exponente utilizado o crea uno por default
   vec<mat<int>> precision;
   if(opts.contains("exp")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "exp"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      precision = clamp_layers_matrix(get_layers_matrix<int, int>(data, capacity), min_exp, max_exp);
   }
   if(precision.empty()) {
      cout << "Trying with default exponent\n";
      precision = full_layer_parameter<int>(capacity, max_exp);
   }
   // Lee las mascaras utilizadas o crea una por default
   vec<mat<bool>> mask;
   if(opts.contains("mask")) {
      path file_path = load_path / format("{}.csv" ,safe_suffix(load_name, "mask"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      mask = get_layers_matrix<bool, int>(data, capacity);
   }
   if(mask.empty()) {
      cout << "Trying with default mask\n";
      mask = full_layer_parameter<bool>(capacity, true);
   }
   // Lee las mascaras utilizadas o crea una por default
   vec<mat<bool>> fixed;
   if(opts.contains("fixed")) {
      path file_path = load_path / format("{}.csv" ,safe_suffix(load_name, "fixed"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      fixed = get_layers_matrix<bool, int>(data, capacity);
   }
   if(fixed.empty()) {
      cout << "Trying with default fixed\n";
      fixed = full_layer_parameter<bool>(capacity, false);
   }
   // Lee los pesos iniciales o crea uno por default
   vec<mat<optional<double>>> init_w;
   if(opts.contains("init")) {
      path file_path = load_path / format("{}.csv", safe_suffix(load_name, "init"));
      const auto& [dim, data] = read_list_from_csv<double>(fstream(file_path), true);
      init_w = get_layers_matrix<optional<double>, double>(data, capacity);
   }
   if(init_w.empty()) {
      cout << "Trying with default init\n";
      init_w = full_layer_parameter<optional<double>>(capacity, optional<double>());
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
   string ResultFile = format("{}.sol{}", file_path, suffix7z);
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
   optional<pair<double, double>> constraint_frac;
   process_yes_arg(opts, "relax_frac", [&constraint_frac](const auto& args) {
      double f,s;
      stringstream(args[0]) >> f;
      stringstream(args[1]) >> s;
      constraint_frac = make_pair(f, s);
   });
   double zero_off_tolerance = 0.0;
   process_yes_arg(opts, "offtol", [&zero_off_tolerance](const auto& args) {
      stringstream(args[0]) >> zero_off_tolerance;
   });
   optional<int> lazy;
   process_yes_arg(opts, "lazy", [&lazy](const auto& args) {
      stringstream(args[0]) >> lazy;
   });
   int samples = 10;
   process_yes_arg(opts, "samples", [&samples](const auto& args) {
      stringstream(args[0]) >> samples;
   });
   optional<double> bound;
   process_yes_arg(opts, "bound", [&bound](const auto& args) {
      stringstream(args[0]) >> bound;
   });
   optional<double> recall;
   process_yes_arg(opts, "recall", [&recall](const auto& args) {
      stringstream(args[0]) >> recall;
   });
   optional<double> slope;
   process_yes_arg(opts, "slope", [&slope](const auto& args) {
      stringstream(args[0]) >> slope;
   });
   optional<pair<double, double>> sparsity;
   process_yes_arg(opts, "sparsity", [&sparsity](const auto& args) {
      double f,s;
      stringstream(args[0]) >> f;
      stringstream(args[1]) >> s;
      sparsity = make_pair(f, s);
   });
   optional<int> lim_bits;
   process_yes_arg(opts, "lim_bits", [&lim_bits](const auto& args) {
      stringstream(args[0]) >> lim_bits;
   });
   optional<double> log_slope;
   process_yes_arg(opts, "log_slope", [&log_slope](const auto& args) {
      stringstream(args[0]) >> log_slope;
   });
   bool use_sos           =  opts.contains("sos"              );
   bool no_shuffle        =  opts.contains("no_shuffle"       );
   bool use_gurobi        =  opts.contains("grb_con"          );
   bool local             =  opts.contains("local"            );
   bool optimize          = !opts.contains("no_opti"          );
   bool save_lp           = !opts.contains("no_lp"            );
   bool save_ilp          = !opts.contains("no_ilp"           );
   bool save_sol          = !opts.contains("no_sol"           );
   bool save_mst          = !opts.contains("no_mst"           );
   bool save_json         = !opts.contains("no_json"          );
   bool no_l1             =  opts.contains("no_l1"            );
   bool no_l2             =  opts.contains("no_l2"            );
   bool use_square        =  opts.contains("square"           );
   bool min_constrs       =  opts.contains("min_constrs"      );
   bool max_constrs       =  opts.contains("max_constrs"      );
   bool restrict          =  opts.contains("restrict"         );
   bool use_bounds        =  opts.contains("use_bounds"       );
   bool use_asym          =  opts.contains("use_asym"         );
   bool minmax            =  opts.contains("minmax"           );
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
   const auto& [ftrs, cls_tgts, reg_tgts, test_ftrs, test_cls_tgts, test_reg_tgts] = (
      sample_data(features, class_targets, regression_targets, samples, !no_shuffle, generator)
   );
   int T = ranges::distance(ftrs);
   cout << "Numero de instancias " << T << "\n";
   // Genera los vectores de desactivar entradas o conexiones en la red neuronal, utilizando una semilla
   // o utilizado de manera aleatoria en caso de no proporcionarla
   process_env(ambiente, opts);
   GRBModel modelo(ambiente);
   NNGRBCallback cb(
      save_path, precision, init_w, mask, ftrs, test_ftrs, bias, activation,
      params, reg_tgts, cls_tgts, test_reg_tgts, test_cls_tgts, reg_loss, cls_loss
   );
   modelo.setCallback(&cb);
   const auto& [b, iw] = gen_model(
      /* GRBEnv ambiente gurobi */modelo,
      /* Numero de casos T */ T,
      /* Numero de capas L */ L,
      /* Capacidad o numero de neuronas por capa */ capacity,
      /* Funciones de activacion */ activation,
      /* Mascaras de pesos */ mask,
      /* Precision para los pesos */ precision,
      /* Numero de bits para los pesos */bits,
      /* Valor inicial o esperado de los pesos */init_w,
      /* Valor de los umbrales o bias */ bias,
      /* Parametros de las funciones de activacion */ params,
      /* Regularización L1 sobre activacion */ l1a,
      /* Regularización L1 sobre pesos */ l1w,
      /* Regularización L2 sobre activacion */ l2a,
      /* Regularización L2 sobre pesos */ l2w,
      /* Penalización por alejarse de la solución inicial */ weight_penalty,
      /* Mantener fijo los pesos */ fixed,
      /* Matriz de las caracteristicas */ ftrs,
      /* Matriz de la regresion esperada */ reg_tgts,
      /* Matriz de la clasificacion deseada */ cls_tgts,
      reg_loss,
      cls_loss,
      /* Tolerancia utilizada en el logaritmo */ zero_tolerance,
      /* Porcentaje de error o tolerancia sobre las restricciones */ constraint_tolerance,
      /* Usar tolerancia para los ceros en las matrices */ zero_off_tolerance,
      /* Prioridad o importancia que se le da más al error que a otras regularizaciones */ error_priority,
      max_constrs,
      min_constrs,
      restrict,
      minmax,
      /* Usar modelo alternativo donde es un offset de los pesos iniciales */ local,
      /* No usar L1 */ no_l1,
      /* No usar L2 */ no_l2,
      /* Usar aproximacion de error cuadratico */ use_square,
      /* Usar restricciones proporcionadas por gurobi */ use_gurobi,
      /* Usar restricciones tipo SOS1 */ use_sos,
      use_bounds,
      use_asym,
      bound,
      sparsity,
      /* Porcentaje de casos usados para restricciones */ constraint_frac,
      /* Tipo de restriccion en gurobi para restricciones no escenciales */ lazy,
      slope,
      recall,
      lim_bits,
      log_slope
   );
   cb.set_binary(b, iw);
   if(save_lp)
      modelo.write(format("{}.lp{}", file_path, suffix7z));
   if(optimize) {
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
               modelo.write(format("{}.sol{}", file_path, suffix7z));
            }
            if(save_json) {
               modelo.write(format("{}.json{}", file_path, suffix7z));
            } 
            if(save_mst) {
               modelo.write(format("{}.mst{}", file_path, suffix7z));
            }
            break;
         case GRB_INFEASIBLE :
            cout << "Modelo infactible\n";
            if(save_ilp) {
               modelo.computeIIS();
               modelo.write(format("{}.ilp{}", file_path, suffix7z));
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
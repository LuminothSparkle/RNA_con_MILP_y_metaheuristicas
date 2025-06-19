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

GRBVar gen_abs_var(
   GRBModel& model, const GRBVar& x,
   const string& var_name, const string& constr_name
) {
   GRBVar var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrAbs(var_abs, x, constr_name);
   return var_abs;
}

template<typename T1, typename T2>
GRBVar gen_diff_var(
   GRBModel& model, const T1& a, const T2& b,
   const string& var_name, const string& constr_name
) {
   GRBVar var_diff = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addConstr(a - b == var_diff, constr_name);
   return var_diff;
}

GRBVar gen_max_var(
   GRBModel& model, const vec<GRBVar>& X, const string& var_name,
   const string& constr_name, double min = 0
) {
   GRBVar var_max = model.addVar(min, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMax(var_max, X.data(), X.size(), min, constr_name);
   return var_max;
}

GRBVar gen_min_var(
   GRBModel& model, const vec<GRBVar>& X, const string& var_name,
   const string& constr_name, double max = 0
) {
   GRBVar var_min = model.addVar(-GRB_INFINITY, max, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMin(var_min, X.data(), X.size(), max, constr_name);
   return var_min;
}

GRBVar gen_sum_var(
   GRBModel& model, const vec<GRBVar>& X,
   const string& var_name, const string& constr_name
) {
   GRBVar var_sum = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   GRBLinExpr expr;
   expr = accumulate(X.begin(),X.end(),expr);
   model.addConstr(expr == var_sum, constr_name);
   return var_sum;
}

template<typename T1, typename T2>
GRBVar gen_abs_error_var(
      GRBModel& model, const T1& y, const T2& ty,
      const string& var_name,  const string& constr_name
   ) {
   GRBVar dy = gen_diff_var(
      model, y, ty,
      format("{}_diff",var_name), 
      format("{}_diff",constr_name)
   );
   return gen_abs_var(model, dy, var_name, constr_name);
}

GRBVar gen_act_var(
      GRBModel& model, const GRBVar& x,
      const string& dropout_name, const optional<double>& dropout = {}
   ) {
   if( dropout.transform([](double dropout) {
      std::random_device rd;
      std::mt19937 gen(rd());   
      return dropout > std::uniform_real_distribution<double>(0.0,1)(gen);
   }).value_or(false) ) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, dropout_name);
   }
   return x;
}

GRBVar gen_bin_w_var(
      GRBModel& model, const GRBVar& b, const GRBVar& a,const string& var_name,
      const string& constr_name, double coef, bool mask
   ) {
   if (!mask) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, format("{}_masked",var_name));
   }
   GRBVar bw = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrIndicator(b, 1, coef * a - bw, GRB_EQUAL, 0, format("{}_on",constr_name));
   model.addGenConstrIndicator(b, 0, bw, GRB_EQUAL, 0, format("{}_off",constr_name));
   return bw;
}

GRBVar gen_bin_var(GRBModel& model, const string& bin_var_name, bool mask = true) {
   if (!mask) {
      return model.addVar(0, 0, 0, GRB_BINARY, bin_var_name);
   }
   return model.addVar(0, 1, 0, GRB_BINARY, bin_var_name);
}

GRBVar gen_hardtanh_var(
      GRBModel& model, const GRBVar& z, const string& var_name,
      const string& constr_name, const pair<double,double>& limits = {-1,1}
   ) {
   GRBVar ht_min = model.addVar(
      -GRB_INFINITY, limits.first, 0,
      GRB_CONTINUOUS, format("{}_min", var_name)
   );
   GRBVar ht = model.addVar(limits.first, limits.second, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMin(ht_min, &z, 1, limits.first, format("{}_min", constr_name));
   model.addGenConstrMax(ht, &ht_min, 1, limits.second, constr_name);
   return ht;
}

GRBVar gen_hardsigmoid_var(
      GRBModel& model, const GRBVar& z,
      const string& var_name, const string& constr_name
   ) {
   GRBVar hs_z = model.addVar(
      -GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS,
      format("{}_z", var_name)
   );
   GRBVar hs_max = model.addVar(
      -GRB_INFINITY, 1, 0, GRB_CONTINUOUS,
      format("{}_max", var_name)
   );
   GRBVar hs = model.addVar(0, 1, 0, GRB_CONTINUOUS, var_name);
   model.addConstr(z / 6 + 0.5 == hs, format("{}_hsin", constr_name));
   model.addGenConstrMin(hs_max, &hs_z, 1, 1, format("{}_min", constr_name));
   model.addGenConstrMax(hs, &hs_max, 1, 0, constr_name);
   return hs;
}

GRBVar gen_ReLU6_var(
      GRBModel& model, const GRBVar& z,
      const string& var_name, const string& constr_name
   ) {
   GRBVar relu6_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("{}_max", var_name));
   GRBVar relu6 = model.addVar(0, 6, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMax(relu6_max, &z, 1, 0, format("{}_max", constr_name));
   model.addGenConstrMin(relu6, &relu6_max, 1, 6, constr_name);
   return relu6;
}

GRBVar gen_ReLU_var(
      GRBModel& model, const GRBVar& z,
      const string& var_name, const string& constr_name
   ) {
   GRBVar relu = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMax(relu, &z, 1, 0, constr_name);
   return relu;
}

GRBVar gen_LeakyReLU_var(
      GRBModel& model, const GRBVar& z, const string& lrelu_var_name,
      const string& lrelu_constr_name, double neg_coef = 0.25
   ) {
   GRBVar lrelu_max = model.addVar(
      -GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS,
      format("{}_max", lrelu_var_name)
   );
   GRBVar lrelu_min = model.addVar(
      -GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS,
      format("{}_min", lrelu_var_name)
   );
   GRBVar lrelu = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, lrelu_var_name);
   model.addGenConstrMax(lrelu_max, &z, 1, 0, format("{}_max", lrelu_constr_name));
   model.addGenConstrMin(lrelu_min, &z, 1, 0, format("{}_min", lrelu_constr_name));
   model.addConstr(lrelu == lrelu_max + neg_coef * lrelu_min, lrelu_constr_name);
   return lrelu;
}

GRBVar gen_act_w_var(
      GRBModel& model, const vec<GRBVar>& bw,
      const string& act_w_var_name, const string& act_w_constr_name
   ) {
   GRBVar aw = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, act_w_var_name);
   model.addConstr(
      accumulate(bw.begin(), bw.end() - 1, GRBLinExpr(-bw.front())),
      GRB_EQUAL, aw, act_w_constr_name
   );
   return aw;
}

vec<GRBVar> gen_input_vars(
      GRBModel& model, const string& var_preffix, const vec<double> fx,
      const optional<double>& dropout = {}
   ) {
   vec<GRBVar> a;
   for(const auto& [i,fx] : views::enumerate(fx)) {
      a.emplace_back(gen_act_var(
         model, 
         model.addVar(fx, fx, 0, GRB_CONTINUOUS, format("{}_{}",var_preffix,i)),
         format("drop{}_{}",var_preffix,i), dropout
      ));
   }
   return a;
}

vec<GRBVar> gen_layer_vars(
      GRBModel& model, const vec<GRBVar>& act, const ten3<GRBVar>& b,
      const GRBVar& bias, const string& var_suffix, const string& constr_suffix, 
      int in_C, int out_C, const mat<int>& precis, 
      const mat<int>& mask, const optional<double>& dropout
   ) {
   vec<GRBVar> a;
   for(const auto& [i, act] : views::enumerate(act)) {
      a.emplace_back(gen_act_var(
         model, act, format("dropz{}_{}",var_suffix,i), dropout
      ));
   }
   a.emplace_back(bias);
   mat<GRBVar> aw( out_C, vec<GRBVar>(in_C + 1) );
   for(const auto& [i,b,a,precis,mask] : views::zip( views::iota(0), b, a, precis, mask)) {
      for(const auto& [j,b,precis,mask] : views::zip( views::iota(0), b, precis, mask )) {
         vec<GRBVar> bw;
         for(const auto& [l,b] : views::enumerate(b)) {
            bw.emplace_back(gen_bin_w_var(
               model, b, a,
               format("bw{}_{}_{}_{}",   var_suffix,    i, j, l),
               format("bw{}_w{},{}_D{}", constr_suffix, i, j, l),
               exp2(l - precis), mask
            ));
         }
         aw[j][i] = gen_act_w_var(
            model, bw,
            format("aw{}_{}_{}",  var_suffix,    i, j),
            format("aw{}_w{},{}", constr_suffix, i, j)
         );
      }
   }
   vec<GRBVar> z;
   for(const auto& [j,aw] : views::enumerate(aw)) {
      z.emplace_back(gen_sum_var(
         model, aw,
         format("z{}_{}",  var_suffix,    j),
         format("z{}_N{}", constr_suffix, j)
      ));
   }
   return z;
}

GRBVar gen_w_var(
      GRBModel& model, const vec<GRBVar>& b, const string& var_name,
      const string& constr_name, int precis = 4
   ) {
   GRBLinExpr expr = -exp2( b.size() - 1.0 - precis ) * b.front();
   for(const auto& [l,b] : views::enumerate(views::take(b, b.size() - 1))) {
      expr += exp2(l - precis) * b;
   }
   GRBVar w = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addConstr(expr == w, constr_name);
   return w;
}

GRBVar gen_l1w_var(
      GRBModel& model, const mat<GRBVar> w,
      const string& var_name, const string& constr_name
   ) {
   GRBLinExpr expr;
   for(double w_i_size = w.size(); const auto& [i, w] : views::enumerate(w)) {
      for(double w_j_size = w.size(); const auto& [j, w] : views::enumerate(w)) {
         expr += 1.0 / (w_i_size * w_j_size) * gen_abs_var(
            model, w, format("{}_{}_{}",var_name,i,j),
            format("{}_w{},{}",constr_name,i,j)
         );
      }
   }
   GRBVar l1w = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addConstr(expr == l1w, constr_name);
   return l1w;
}

GRBVar gen_l1a_var(
      GRBModel& model, const vec<GRBVar>& a,
      const string& var_name, const string& constr_name
   ) {
   GRBLinExpr expr;
   GRBVar l1a = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   for(double coef = 1.0 / a.size(); const auto& [j,a] : views::enumerate(a)) {
      expr += coef * gen_abs_var(
         model, a,
         format("{}_{}",var_name,j),
         format("{}_N{}",constr_name,j)
      );
   }
   model.addConstr(expr == l1a, constr_name);
   return l1a;
}

GRBVar gen_class_error_var(
      GRBModel& model, const vec<GRBVar>& y, const string& var_name,
      const string& constr_name, const vec<double>& ty,
      double zero_tolerance = 0.0001
   ) {
   vec<tuple<double,GRBVar,int>> c_order;
   for(const auto& tup : views::zip( ty, y, views::iota(0) )) {
      c_order.push_back(tup);
   }
   std::sort( c_order.begin(), c_order.end(), [](const auto& tuple_a, const auto& tuple_b) {
      return get<double>(tuple_a) < get<double>(tuple_b);
   } );
   const auto& pred = [zero_tolerance](const auto& tuple) {
      const auto& [ty,y,j] = tuple;
      return ty <= zero_tolerance;
   };
   GRBLinExpr expr;
   for(const auto& [ty,y,j] : views::take_while(c_order, pred)) {
      expr += y;
   }
   auto non_zero = views::drop_while(c_order, pred);
   for(const auto& [slide,tuple] : views::enumerate(non_zero)) {
      const auto& [tc_a,y_a,j_a] = tuple;
      for(const auto&[tc_b,y_b,j_b] : views::drop(non_zero, slide + 1)) {
         GRBVar constrvio = model.addVar(
            -GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS,
            format("{}_{}_{}_constrvio", var_name, j_a, j_b)
         );
         model.addConstr(
            y_a - y_b + constrvio == log(tc_a / tc_b),
            format("{}_Rel{},{}", constr_name, j_a, j_b)
         );
         expr += gen_abs_var(
            model, constrvio,
            format("{}_{}_{}_abs", var_name,    j_a, j_b),
            format("{}_Err{},{}",  constr_name, j_a, j_b)
         );
      }
   }
   GRBVar EC = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addConstr(expr == EC, constr_name);
   return EC;
}

GRBVar gen_regression_error_var(
      GRBModel& model, const vec<GRBVar>& y, const string& var_name,
      const string& constr_name, const vec<double>& ty
   ) {
   GRBLinExpr expr;
   for(const auto& [i,y,ty] : views::zip( views::iota(0), y, ty )) {
      expr += gen_abs_error_var(
         model, y, ty,
         format("{}_{}",  var_name,    i),
         format("{}_N{}", constr_name, i)
      );
   }
   GRBVar ER = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addConstr(expr == ER, constr_name);
   return ER;
}

vec<GRBVar> gen_activation_vars(
   GRBModel& model, const string& type, const vec<GRBVar>& z,
   const string& var_suffix, const string constr_suffix,
   const vec<double>& LeakyReLU_coef, const pair<double,double>& hardtanh_limits
) {
   vec<GRBVar> a;
   if(type == "ReLU") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_ReLU_var(
            model, z,
            format("relu{}_{}",  var_suffix,    j),
            format("relu{}_N{}", constr_suffix, j)
         ));
      }
   }
   else if(type == "ReLU6") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_ReLU6_var(
            model, z,
            format("relu6{}_{}",  var_suffix,    j),
            format("relu6{}_N{}", constr_suffix, j)
         ));
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_LeakyReLU_var(
            model, z, format("Lrelu{}_{}",var_suffix,j),
            format("Lrelu{}_N{}",constr_suffix,j), LeakyReLU_coef[j]
         ));
      }
   }
   else if(type  == "Hardtanh") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_hardtanh_var(
            model, z, format("ht{}_{}",var_suffix,j),
            format("ht{}_N{}",constr_suffix,j), hardtanh_limits
         ));
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_hardsigmoid_var(
            model, z,
            format("hs{}_{}",  var_suffix,    j),
            format("hs{}_N{}", constr_suffix, j)
         ));
      }
   }
   else {
      a = z;
   } 
   return a;
}

GRBModel get_model(
      const GRBEnv& environment, int T, int L, const vec<int>& C, const vec<string>& AF, 
      const vec<mat<int>>& mask, const vec<mat<int>>& precis, const vec<mat<int>>& D,
      const vec<double>& bias_w, const mat<double>& leakyReLU_coef,
      const vec<pair<double,double>>& hardtanh_limits,
      const vec<optional<double>>& dropout,
      const vec<optional<double>>& l1a_norm,
      const vec<optional<double>>& l1w_norm,
      const mat<double>& fx, const mat<double>& reg_ty, const mat<double>& class_ty,
      double zero_tolerance = 0.0001
   ) {
   GRBModel model(environment);
   ten4<GRBVar> b(L);
   vec<GRBVar> bias(L);
   GRBLinExpr L1_expr;
   for(int k = 0; k < L; ++k) {
      b[k] = ten3<GRBVar>(C[k] + 1,mat<GRBVar>(C[k + 1]));
      mat<GRBVar> w(C[k] + 1,vec<GRBVar>(C[k + 1]));
      for(int i = 0; i <= C[k]; ++i) {
         for(int j = 0; j < C[k + 1]; ++j) {
            b[k][i][j] = vec<GRBVar>(D[k][i][j] + 1);
            for(int l = 0; l <= D[k][i][j]; ++l) {
               b[k][i][j][l] = gen_bin_var(model, format("b_{}_{}_{}_{}",k,i,j,l), mask[k][i][j]);
            }
            w[i][j] = gen_w_var(
               model, b[k][i][j], format("w_{}_{}_{}",k,i,j),
               format("w_{}_{}_{}",k,i,j), precis[k][i][j]
            );
         }
      }
      if(l1w_norm[k]) {
         L1_expr += *l1w_norm[k] * gen_l1w_var(model, w, format("l1w_{}",k), format("l1w_L{}",k));
      }
      bias[k] = model.addVar(bias_w[k], bias_w[k], 0, GRB_CONTINUOUS, format("bias_{}",k));
   }
   GRBLinExpr EC_expr, ER_expr;
   for(const auto& [t,fx,tc_y,treg_y] : views::zip(views::iota(0),fx,class_ty,reg_ty)) {
      vec<GRBVar> a = gen_input_vars(model, format("x_{}",t), fx, dropout.back());
      for(const auto& [k, b, AF, D, precis, bias, dropout, leakyReLU_coef, hardtanh_limits] : views::zip(
         views::iota(0), b, AF, D,precis, bias, views::drop(dropout,1), leakyReLU_coef, hardtanh_limits
      )) {
         const auto& z = gen_layer_vars(
            model, a, b, bias,
            format("_{}_{}",t,k), format("_L{}_C{}",k,t),
            C[k], C[k + 1], precis,
            mask[k], dropout
         );
         a = gen_activation_vars(
            model, AF, z, format("_{}_{}",t,k),
            format("_L{}_C{}",k,t), leakyReLU_coef, hardtanh_limits
         );
         if(l1a_norm[k]) {
            L1_expr += *l1a_norm[k] * gen_l1a_var(
               model, a, format("l1a_{}_{}",t,k), format("l1a_L{}_C{}",k,t)
            );
         }
      }
      const auto& ry_view = views::take(a, treg_y.size());
      const auto& cy_view = views::drop(a, treg_y.size());
      EC_expr += gen_class_error_var(
         model, vec<GRBVar>( ry_view.begin(), ry_view.end() ),
         format("EC_{}",t), format("ClassE_{}",t),
         treg_y, zero_tolerance
      );
      ER_expr += gen_regression_error_var(
         model, vec<GRBVar>( cy_view.begin(), cy_view.end() ),
         format("ER_{}",t), format("RegE_{}",t),
         tc_y
      );
   }
   model.setObjective(EC_expr + ER_expr + L1_expr, GRB_MINIMIZE);
   return model;
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
   line_stream.str(line);
   if(ignore_index) {
      getline(line_stream, word);
   }
   int max_dim = 0;
   while(getline(line_stream, word) && word.starts_with("d_")) {
      ++max_dim;
   }
   vec<vec<int>> dim_list;
   vec<vec<T>> data_list;
   while(getline(input, line)) {
      line_stream.str(line);
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
   vec<optional<double>> Drop, L1w, L1a;
   while(getline(input, line)) {
      line_stream = stringstream(line);
      if(ignore_index) {
         getline(line_stream, word, ',');
      }
      int k;
      optional<double> l1a, l1w, drop, b, ht_min, ht_max;
      optional<string> af;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> k;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> af;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> drop;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> ht_min;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> ht_max;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> l1w;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> l1a;
      getline(line_stream, word, ','); word_stream = stringstream(word); word_stream >> b;
      C.emplace_back(k);
      AF.emplace_back(af.value_or("None"));
      HT.emplace_back(make_pair(ht_min.value_or(-1),ht_max.value_or(1)));
      Drop.emplace_back(drop);
      L1w.emplace_back(l1w);
      L1a.emplace_back(l1a);
      bias.emplace_back(b.value_or(1.0));
   }
   return make_tuple(C, AF, HT, Drop, L1w, L1a, bias);
}

template<typename T>
auto get_layers_matrix(const mat<int>& dim, const vec<vec<T>>& data) {
   vec<mat<T>> layers_data;
   for(const auto& [dim,data] : views::zip(dim,data)) {
      int n = dim[0], m = dim[1];
      mat<T> layer_data(n,vec<T>(m));
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
      {"zero_tolerance", 1}
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
   path arch_path = load_path / format("{}.csv",safe_suffix(load_name,"arch"));
   path features_path = load_path / format("{}.csv",safe_suffix(load_name,"ftr"));
   path class_targets_path = load_path / format("{}.csv",safe_suffix(load_name,"cls_tgt"));
   path regression_targets_path = load_path / format("{}.csv",safe_suffix(load_name,"reg_tgt"));
   const auto& [C, AF, hardtanh, dropout, l1w_norm, l1a_norm, bias] = read_arch(fstream(arch_path));
   const auto& features = read_matrix_from_csv<double>(fstream(features_path));
   const auto& regression_targets = read_matrix_from_csv<double>(fstream(regression_targets_path));
   const auto& class_targets = read_matrix_from_csv<double>(fstream(class_targets_path));
   int T = features.size(), L = C.size() - 1;
   vec<mat<int>> bits;
   if(opts.contains("use_bits")) {
      path file_path = load_path / format("{}.csv",safe_suffix(load_name,"bits"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path.string()));
      bits = get_layers_matrix(dim,data);
   }
   else {
      bits = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         bits[k] = mat<int>(C[k] + 1,vec<int>(C[k + 1]));
         for(auto& row : bits[k]) {
            for(auto& value : row) {
               value = 8;
            }
         }
      }
   }
   vec<mat<int>> precision;
   if(opts.contains("use_precision")) {
      path file_path = load_path / format("{}.csv",safe_suffix(load_name,"precision"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path.string()));
      precision = get_layers_matrix(dim,data);
   }
   else {
      precision = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         precision[k] = mat<int>(C[k] + 1,vec<int>(C[k + 1]));
         for(auto& row : precision[k]) {
            for(auto& value : row) {
               value = 4;
            }
         }
      }
   }
   vec<mat<int>> mask;
   if(opts.contains("use_mask")) {
      path file_path = load_path / format("{}.csv",safe_suffix(load_name,"mask"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      mask = get_layers_matrix(dim,data);
   }
   else {
      mask = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         mask[k] = mat<int>(C[k] + 1,vec<int>(C[k + 1]));
         for(auto& row : mask[k]) {
            for(auto& value : row) {
               value = 1;
            }
         }
      }
   }
   vec<vec<double>> leakyReLU;
   if(opts.contains("use_leakyReLU")) {
      path file_path = load_path / format("{}.csv",safe_suffix(load_name,"lReLU"));
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
   string file_path = (save_path / safe_suffix(load_name,save_name)).string();
   string ResultFile = format("{}.sol",file_path);
   string SolFiles = file_path;
   string LogFile = format("{}.log",file_path);
   GRBEnv ambiente;
   stringstream argstream;
   int LogToConsole = !opts.contains("no_log_to_console");
   ambiente.set(GRB_IntParam_LogToConsole, LogToConsole);
   process_no_arg(opts, "no_save_sols", [&SolFiles,&ambiente]() {
      ambiente.set(GRB_StringParam_SolFiles, SolFiles);
   });
   process_no_arg(opts, "no_save_log", [&LogFile,&ambiente]() {
      ambiente.set(GRB_StringParam_LogFile, LogFile);
   });
   process_yes_arg(opts, "best_obj_stop", [&argstream,&ambiente](const auto& args) {
      double BestObjStop;
      argstream.str(args[0]);
      argstream >> BestObjStop;
      ambiente.set(GRB_DoubleParam_BestObjStop, BestObjStop);
   });
   double zero_tolerance = 0.001;
   process_yes_arg(opts, "zero_tolerance", [&argstream,&zero_tolerance](const auto& args) {
      argstream.str(args[0]);
      argstream >> zero_tolerance;
   });
   process_yes_arg(opts, "feas_tol", [&argstream,&ambiente](const auto& args) {
      double FeasibilityTol;
      argstream.str(args[0]);
      argstream >> FeasibilityTol;
      ambiente.set(GRB_DoubleParam_FeasibilityTol, FeasibilityTol);
   });
   process_yes_arg(opts, "int_feas_tol", [&argstream,&ambiente](const auto& args) {
      double IntFeasTol;
      argstream.str(args[0]);
      argstream >> IntFeasTol;
      ambiente.set(GRB_DoubleParam_IntFeasTol, IntFeasTol);
   });
   process_yes_arg(opts, "iteration_limit", [&argstream,&ambiente](const auto& args) {
      double IterationLimit;
      argstream.str(args[0]);
      argstream >> IterationLimit;
      ambiente.set(GRB_DoubleParam_IterationLimit, IterationLimit);
   });
   process_yes_arg(opts, "opt_tol", [&argstream,&ambiente](const auto& args) {
      double OptimalityTol;
      argstream.str(args[0]);
      argstream >> OptimalityTol;
      ambiente.set(GRB_DoubleParam_OptimalityTol, OptimalityTol);
   });
   process_yes_arg(opts, "solution_limit", [&argstream,&ambiente](const auto& args) {
      int SolutionLimit;
      argstream.str(args[0]);
      argstream >> SolutionLimit;
      ambiente.set(GRB_IntParam_SolutionLimit, SolutionLimit);
   });
   process_yes_arg(opts, "time_limit", [&argstream,&ambiente](const auto& args) {
      double TimeLimit;
      argstream.str(args[0]);
      argstream >> TimeLimit;
      ambiente.set(GRB_DoubleParam_TimeLimit, TimeLimit);
   });
   process_yes_arg(opts, "node_limit", [&argstream,&ambiente](const auto& args) {
      double NodeLimit;
      argstream.str(args[0]);
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

   // construcción del modelo
   GRBModel modelo = get_model(
      ambiente, T, L, C, AF,
      mask, precision, bits, bias,
      leakyReLU, hardtanh, dropout, l1a_norm, l1w_norm,
      features, class_targets, regression_targets,
      zero_tolerance
   );
   // ------ resolución del modelo
   if(save_lp)
      modelo.write(path(format("{}.lp",file_path)).string());
   if(optimize) {
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
               modelo.write(format("{}.sol",file_path));
            }
            if(save_json) {
               modelo.write(format("{}.json",file_path));
            } 
            if(save_mst) {
               modelo.write(format("{}.mst",file_path));
            }
            break;
         case GRB_INFEASIBLE :
            cout << "Modelo infactible\n";
            if(save_ilp) {
               modelo.computeIIS( );
               modelo.write(format("{}.ilp",file_path));
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
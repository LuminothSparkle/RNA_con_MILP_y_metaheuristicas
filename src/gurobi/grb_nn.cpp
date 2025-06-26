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

bool is_single_expr(const GRBLinExpr& expr) {
   return expr.size() == 1 && expr.getConstant() == 0.0 && expr.getCoeff(0) == 1.0;
}

GRBVar gen_var(
   GRBModel& model, const GRBLinExpr& expr,
   const string& var_name, const string& constr_name,
   double min = -GRB_INFINITY, double max = GRB_INFINITY,
   char var_type = GRB_CONTINUOUS
) {
   if(is_single_expr(expr)) {
      return expr.getVar(0);
   }
   GRBVar var = model.addVar(min, max, 0, var_type, var_name);
   model.addConstr(expr == var, constr_name);
   return var;
}

vec<GRBVar> gen_vars(
   GRBModel& model, const vec<GRBLinExpr>& exprs,
   const string& var_name, const string& constr_name,
   double min = -GRB_INFINITY, double max = GRB_INFINITY,
   char var_type = GRB_CONTINUOUS
) {
   vec<GRBVar> vars(exprs.size());
   for(auto&& [i,var,expr] : views::zip(views::iota(0),vars,exprs)) {
      var = gen_var(
         model,expr,
         format("{}_{}",var_name,i),format("{}_{}",constr_name,i),
         min,max,var_type
      );
   }
   return vars;
}

GRBVar gen_abs_var(
   GRBModel& model, const GRBLinExpr& x,
   const string& var_name, const string& constr_name,
   bool use_max, bool objective
);

GRBVar gen_abs_var(
   GRBModel& model, const GRBVar& x,
   const string& var_name, const string& constr_name,
   bool use_max = false, bool objective = false
) {
   if(objective || use_max) {
      return gen_abs_var(
         model, GRBLinExpr(gen_var(
            model, x,
            format("{}_input",var_name),
            format("{}_input",constr_name)
         )),
         var_name, constr_name,
         use_max, objective
      );
   }
   GRBVar var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrAbs(var_abs, x, constr_name);
   return var_abs;
}

GRBVar gen_abs_var(
   GRBModel& model, const GRBLinExpr& x,
   const string& var_name, const string& constr_name,
   bool use_max = false, bool objective = false
) {
   if(objective || use_max) {
      GRBVar var_abs = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
      model.addConstr(x <= var_abs, format("{}_plus",constr_name));
      model.addConstr(-x <= var_abs, format("{}_minus",constr_name));
      if(!objective) {
         GRBVar a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or",var_name));
         model.addGenConstrIndicator(a_or_b, 1, var_abs - x, GRB_GREATER_EQUAL, 0, format("{}_on",constr_name));
         model.addGenConstrIndicator(a_or_b, 0, var_abs + x, GRB_GREATER_EQUAL, 0, format("{}_off",constr_name));
      }
      return var_abs;
   }
   return gen_abs_var(
      model, gen_var(
         model, x,
         format("{}_input",var_name),
         format("{}_input",constr_name)
      ),
      var_name, constr_name,
      use_max, objective
   );
}

template<typename T>
GRBLinExpr gen_abs_expr(
   GRBModel& model, const T& x,
   const string& var_name, const string& constr_name,
   bool use_max = false, bool objective = false
) {
   return gen_abs_var(model,x,var_name,constr_name,use_max, objective);
}

GRBVar gen_max_var(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name
) {
   GRBVar var_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   GRBVar a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or",var_name));
   model.addConstr(var_max >= a, format("{}_a",constr_name));
   model.addConstr(var_max >= b, format("{}_b",constr_name));
   model.addGenConstrIndicator(a_or_b, 1, var_max - a, GRB_LESS_EQUAL, 0, format("{}_on",constr_name));
   model.addGenConstrIndicator(a_or_b, 0, var_max - b, GRB_LESS_EQUAL, 0, format("{}_off",constr_name));
   return var_max;
}

GRBLinExpr gen_max_expr(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name
) {
   return gen_max_var(
      model,
      a,b,
      var_name,
      constr_name
   );
}

GRBVar gen_min_var(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name
) {
   GRBVar var_min = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   GRBVar a_or_b = model.addVar(0, 1, 0, GRB_BINARY, format("{}_or",var_name));
   model.addConstr(var_min <= a, format("{}_a",constr_name));
   model.addConstr(var_min <= b, format("{}_b",constr_name));
   model.addGenConstrIndicator(a_or_b, 1, var_min - a, GRB_GREATER_EQUAL, 0, format("{}_on",constr_name));
   model.addGenConstrIndicator(a_or_b, 0, var_min - b, GRB_GREATER_EQUAL, 0, format("{}_off",constr_name));
   return var_min;
}

GRBLinExpr gen_min_expr(
   GRBModel& model, const GRBLinExpr& a, const GRBLinExpr& b,
   const string& var_name, const string& constr_name
) {
   return gen_min_var(
      model,
      a,b,
      var_name,
      constr_name
   );
}

GRBVar gen_max_var(
   GRBModel& model, const vec<GRBVar>& X,
   const string& var_name, const string& constr_name,
   double min = 0
) {
   GRBVar var = model.addVar(min, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMax(var, X.data(), X.size(), min, constr_name);
   return var;
}

GRBVar gen_max_var(
   GRBModel& model, const GRBLinExpr& X,
   const string& var_name, const string& constr_name,
   double min = 0
) {
   vec<GRBVar> X_vec = {
      gen_var(
         model, X,
         format("{}_input",var_name),
         format("{}_input",constr_name)
      )
   };
   return gen_max_var(model, X_vec, var_name, constr_name, min);
}

GRBLinExpr gen_max_expr(
   GRBModel& model, const vec<GRBVar>& X, const string& var_name,
   const string& constr_name, double min = 0
) {
   return gen_max_var(model,X,var_name,constr_name,min);
}

GRBLinExpr gen_max_expr(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double min = 0
) {
   return gen_max_var(
      model,
      gen_vars(
         model,X,
         format("{}_inputs",var_name),
         format("{}_inputs",constr_name)
      ),
      var_name,constr_name,min
   );
}

GRBVar gen_max_var(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double min = 0
) {
   return gen_max_var(
      model,
      gen_vars(
         model,X,
         format("{}_inputs",var_name),
         format("{}_inputs",constr_name)
      ),
      var_name,constr_name,min
   );
}

GRBVar gen_min_var(
   GRBModel& model, const vec<GRBVar>& X,
   const string& var_name, const string& constr_name,
   double max = 0
) {
   GRBVar var = model.addVar(-GRB_INFINITY, max, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrMin(var, X.data(), X.size(), max, constr_name);
   return var;
}

GRBVar gen_min_var(
   GRBModel& model, const GRBLinExpr& X,
   const string& var_name, const string& constr_name,
   double max = 0
) {
   vec<GRBVar> X_vec = {
      gen_var(
         model, X,
         format("{}_input",var_name),
         format("{}_input",constr_name)
      )
   };
   return gen_min_var(model, X_vec, var_name, constr_name, max);
}

GRBLinExpr gen_min_expr(
   GRBModel& model, const vec<GRBVar>& X, const string& var_name,
   const string& constr_name, double max = 0
) {
   return gen_min_var(model,X,var_name,constr_name,max);
}

GRBLinExpr gen_min_expr(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double max = 0
) {
   return gen_min_expr(
      model,
      gen_vars(
         model,X,
         format("{}_inputs",var_name),
         format("{}_inputs",constr_name)
      ),
      var_name,constr_name,max
   );
}

GRBVar gen_min_var(
   GRBModel& model, const vec<GRBLinExpr>& X, const string& var_name,
   const string& constr_name, double max = 0
) {
   return gen_min_var(
      model,
      gen_vars(
         model,X,
         format("{}_inputs",var_name),
         format("{}_inputs",constr_name)
      ),
      var_name,constr_name,max
   );
}

template<typename T>
GRBLinExpr gen_sum_expr(const vec<T>& X) {
   return accumulate(X.begin(),X.end(),GRBLinExpr());
}

template<typename T>
GRBVar gen_sum_var(
   GRBModel& model, const vec<T>& X,
   const string& var_name, const string& constr_name
) {
   return gen_var(model,gen_sum_expr(X),var_name,constr_name);
}

GRBLinExpr gen_abs_error_expr(
   GRBModel& model, const GRBLinExpr& y, const GRBLinExpr& ty,
   const string& var_name,  const string& constr_name,
   bool use_max = false, bool objective = false
) {
   return gen_abs_expr(model, y - ty, var_name, constr_name, use_max,objective);
}

GRBVar gen_abs_error_var(
   GRBModel& model, const GRBLinExpr& y, const GRBLinExpr& ty,
   const string& var_name,  const string& constr_name,
   bool use_max = false, bool objective = false
) {
   return gen_var(
      model,
      gen_abs_error_expr(
         model, y, ty, 
         format("{}_input",var_name),
         format("{}_input",constr_name),
         use_max,objective
      ),
      var_name, constr_name, 0
   );
}

template<typename T>
GRBVar gen_act_var(
   GRBModel& model, const T& x, const string& dropout_name,
   const optional<double>& dropout = {}
) {
   if( dropout.transform([](double dropout) {
      std::random_device rd;
      std::mt19937 gen(rd());   
      return dropout > std::uniform_real_distribution<double>(0.0,1)(gen);
   }).value_or(false) ) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, dropout_name);
   }
   return gen_var(
      model,x,
      format("{}_output",dropout_name),
      format("{}_output",dropout_name)
   );
}

template<typename T>
GRBLinExpr gen_act_expr(
   GRBModel& model, const T& x, const string& dropout_name,
   const optional<double>& dropout = {}
) {
   if( dropout.transform([](double dropout) {
      std::random_device rd;
      std::mt19937 gen(rd());   
      return dropout > std::uniform_real_distribution<double>(0.0,1)(gen);
   }).value_or(false) ) {
      return GRBLinExpr();
   }
   return gen_var(
      model,x,
      format("{}_output",dropout_name),
      format("{}_output",dropout_name)
   );
}

GRBVar gen_bin_w_var(
   GRBModel& model, const GRBVar& b, const GRBLinExpr& a,
   const string& var_name, const string& constr_name,
   double coef, bool mask
) {
   if (!mask) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, format("{}_masked",var_name));
   }
   GRBVar bw = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, var_name);
   model.addGenConstrIndicator(b, 1, coef * a - bw, GRB_EQUAL, 0, format("{}_on",constr_name));
   model.addGenConstrIndicator(b, 0, bw, GRB_EQUAL, 0, format("{}_off",constr_name));
   return bw;
}

GRBVar gen_bin_w_var(
   GRBModel& model, const GRBLinExpr& b, const GRBLinExpr& a,
   const string& var_name, const string& constr_name,
   double coef, bool mask
) {
   if (!mask) {
      return model.addVar(0, 0, 0, GRB_CONTINUOUS, format("{}_masked",var_name));
   }
   GRBVar b_var = gen_var(
      model,b,
      format("{}_input",var_name),
      format("{}_input",constr_name),
      0,mask,
      GRB_BINARY
   );
   return gen_bin_w_var(model,b_var,a,var_name,constr_name,coef,mask);
}

GRBLinExpr gen_bin_w_expr(
   GRBModel& model, const GRBLinExpr& b, const GRBLinExpr& a,
   const string& var_name, const string& constr_name,
   double coef, bool mask
) {
   if (!mask) {
      return GRBLinExpr();
   }
   GRBVar b_var = gen_var(
      model,b,
      format("{}_input",var_name),
      format("{}_input",constr_name),
      0,mask,
      GRB_BINARY
   );
   return gen_bin_w_var(model,b_var,a,var_name,constr_name,coef,mask);
}

GRBVar gen_bin_var(GRBModel& model, const string& var_name, bool mask = true) {
   return model.addVar(0, mask, 0, GRB_BINARY, var_name);
}

GRBLinExpr gen_bin_expr(GRBModel& model, const string& var_name, bool mask = true) {
   if (!mask) {
      return GRBLinExpr();
   }
   return gen_bin_var(model,var_name,mask);
}

GRBVar gen_hardtanh_var(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name,
   const pair<double,double>& limits = {-1,1}
) {
   GRBVar ht_min = gen_min_var(
      model,z,
      limits.second,
      format("{}_min",var_name),
      format("{}_min",constr_name)
   );
   return gen_max_var(model,ht_min,limits.first,var_name,constr_name);
}

GRBVar gen_hardtanh_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name,
   const pair<double,double>& limits = {-1,1}
) {
   return gen_hardtanh_var(
      model,
      gen_var(
         model,z,
         format("{}_input",var_name),
         format("{}_input",constr_name)
      ),
      var_name,constr_name,limits
   );
}

template<typename T>
GRBLinExpr gen_hardtanh_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name,
   const pair<double,double>& limits = {-1,1}
) {
   return gen_hardtanh_var(model,z,var_name,constr_name,limits);
}

GRBVar gen_hardsigmoid_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name
) {
   GRBVar hs_min = gen_min_var(
      model,
      z / 6 + 0.5,1,
      format("{}_min",var_name),
      format("{}_min",constr_name)
   );
   return gen_max_var(model,hs_min,0,var_name,constr_name);
}

template<typename T>
GRBLinExpr gen_hardsigmoid_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name
) {
   return gen_hardsigmoid_var(model,z,var_name,constr_name);
}

GRBVar gen_ReLU6_var(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name
) {
   GRBVar relu6_max = gen_max_var(
      model,
      z,0,
      format("{}_max",var_name),
      format("{}_max",constr_name)
   );
   return gen_min_var(model,relu6_max,6,var_name,constr_name);
}

GRBVar gen_ReLU6_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name
) {
   return gen_ReLU6_var(
      model,
      gen_var(
         model,z,
         format("{}_input",var_name),
         format("{}_input",constr_name)
      ),
      var_name,constr_name
   );
}

template<typename T>
GRBLinExpr gen_ReLU6_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name
) {
   return gen_ReLU6_var(model,z,var_name,constr_name);
}

GRBVar gen_ReLU_var(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name
) {
   return gen_max_var(model,z,0,var_name,constr_name);
}

GRBVar gen_ReLU_var(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name
) {
   return gen_ReLU_var(
      model,
      gen_var(
         model, z,
         format("{}_input",var_name),
         format("{}_input",constr_name)
      ),
      var_name,constr_name
   );
}

template<typename T>
GRBLinExpr gen_ReLU_expr(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name
) {
   return gen_ReLU_var(model,z,var_name,constr_name);
}

GRBLinExpr gen_LeakyReLU_expr(
   GRBModel& model, const GRBVar& z,
   const string& var_name, const string& constr_name,
   double neg_coef = 0.25
) {
   GRBVar lrelu_max = gen_max_var(
      model,
      z,0,
      format("{}_max", var_name),
      format("{}_max", constr_name)
   );
   GRBVar lrelu_min = gen_min_var(
      model,
      z,0,
      format("{}_min", var_name),
      format("{}_min", constr_name)
   );
   return lrelu_max + neg_coef * lrelu_min;
}

GRBLinExpr gen_LeakyReLU_expr(
   GRBModel& model, const GRBLinExpr& z,
   const string& var_name, const string& constr_name,
   double neg_coef = 0.25
) {
   return gen_LeakyReLU_expr(
      model,
      gen_var(
         model,z,
         format("{}_input",var_name),
         format("{}_input",constr_name)
      ),
      var_name,constr_name,neg_coef
   );
}

template<typename T>
GRBVar gen_LeakyReLU_var(
   GRBModel& model, const T& z,
   const string& var_name, const string& constr_name,
   double neg_coef = 0.25
) {
   return gen_var(
      model,
      gen_LeakyReLU_expr(model,z,var_name,constr_name,neg_coef),
      format("{}_output",var_name),
      format("{}_output",constr_name)
   );
}

template<typename T>
GRBLinExpr gen_act_w_expr(const vec<T>& bw) {
   return accumulate(bw.begin(), bw.end() - 1, GRBLinExpr(-bw.front()));
}

template<typename T>
GRBVar gen_act_w_var(
   GRBModel& model, const vec<T>& bw,
   const string& var_name, const string& constr_name
) {
   return gen_var(
      model,
      gen_act_w_expr(bw),
      var_name, constr_name
   );
}

vec<GRBLinExpr> gen_input_exprs(
   GRBModel& model, const string& var_name,
   const vec<double> fx, const optional<double>& dropout = {}
) {
   vec<GRBLinExpr> a;
   for(const auto& [i,fx] : fx | views::enumerate) {
      a.emplace_back(gen_act_expr(
         model, 
         model.addVar(fx, fx, 0, GRB_CONTINUOUS, format("{}_{}",var_name,i)),
         format("drop{}_{}",var_name,i),
         dropout
      ));
   }
   return a;
}

vec<GRBVar> gen_input_vars(
   GRBModel& model, const string& var_name,
   const vec<double> fx, const optional<double>& dropout = {}
) {
   return gen_vars(
      model,
      gen_input_exprs(model,var_name,fx,dropout),
      format("{}_output",var_name),
      format("{}_output",var_name)
   );
}

template<typename T1, typename T2>
vec<GRBLinExpr> gen_layer_exprs(
   GRBModel& model, const vec<T1>& act, const ten3<T2>& b,
   const GRBLinExpr& bias, const string& var_name, const string& constr_name, 
   int in_C, int out_C, const mat<int>& precis, 
   const mat<int>& mask, const optional<double>& dropout
) {
   vec<GRBLinExpr> a;
   for(const auto& [i, act] : act | views::enumerate) {
      a.emplace_back(gen_act_expr(
         model, act,
         format("dropz{}_{}",var_name,i),
         dropout
      ));
   }
   a.emplace_back(bias);
   mat<GRBLinExpr> aw( out_C, vec<GRBLinExpr>(in_C + 1) );
   for(const auto& [i,b,a,precis,mask] : views::zip( views::iota(0), b, a, precis, mask)) {
      for(const auto& [j,b,precis,mask] : views::zip( views::iota(0), b, precis, mask )) {
         vec<GRBLinExpr> bw;
         for(const auto& [l,b] : b | views::enumerate) {
            
            bw.emplace_back(gen_bin_w_expr(
               model,
               gen_var(
                  model,b,
                  format("{}_input",var_name),
                  format("{}_input",constr_name),
                  0,mask,
                  GRB_BINARY
               ),
               a,
               format("bw{}_{}_{}_{}",   var_name,    i, j, l),
               format("bw{}_w{},{}_D{}", constr_name, i, j, l),
               exp2(l - precis), mask
            ));
         }
         aw[j][i] = gen_act_w_expr(bw);
      }
   }
   vec<GRBLinExpr> z;
   for(const auto& [j,aw] : aw | views::enumerate) {
      z.emplace_back(gen_sum_expr(aw));
   }
   return z;
}

template<typename T>
vec<GRBVar> gen_layer_vars(
   GRBModel& model, const vec<T>& act, const ten3<GRBVar>& b,
   const GRBVar& bias, const string& var_name, const string& constr_name, 
   int in_C, int out_C, const mat<int>& precis, 
   const mat<int>& mask, const optional<double>& dropout
) {
   return gen_vars(
      model,
      gen_layer_exprs(
         model,act,b,bias,
         var_name,constr_name,
         in_C,out_C,
         precis,mask,dropout
      ),
      format("{}_outputs",var_name),
      format("{}_outputs",constr_name)
   );
}

template<typename T>
GRBLinExpr gen_w_expr(
   GRBModel& model, const vec<T>& b, int precis = 4
) {
   GRBLinExpr expr = -exp2( b.size() - 1.0 - precis ) * b.front();
   for(const auto& [l,b] : b | views::take(b.size() - 1) | views::enumerate) {
      expr += exp2(l - precis) * b;
   }
   return expr;
}

template<typename T>
GRBVar gen_w_var(
   GRBModel& model, const vec<T>& b, const string& var_name,
   const string& constr_name, int precis = 4
) {
   return gen_var(
      model,gen_w_expr(model,b,precis),
      var_name,constr_name,precis
   );
}

template<typename T>
GRBLinExpr gen_l1w_expr(
   GRBModel& model, const mat<T> w,
   const string& var_name, const string& constr_name
) {
   GRBLinExpr expr;
   for(double w_i_size = w.size(); const auto& [i, w] : w | views::enumerate) {
      for(double w_j_size = w.size(); const auto& [j, w] : w | views::enumerate) {
         expr += 1.0 / (w_i_size * w_j_size) * gen_abs_expr(
            model, w,
            format("{}_{}_{}",var_name,i,j),
            format("{}_w{},{}",constr_name,i,j),
            true,true
         );
      }
   }
   return expr;
}

template<typename T>
GRBVar gen_l1w_var(
   GRBModel& model, const mat<T> w,
   const string& var_name, const string& constr_name
) {
   return gen_var(
      model,gen_l1w_expr(model,w,var_name,constr_name),
      format("{}_output",var_name),
      format("{}_output",constr_name),
      0
   );
}

template<typename T>
GRBLinExpr gen_l1a_expr(
   GRBModel& model, const vec<T>& a,
   const string& var_name, const string& constr_name
) {
   GRBLinExpr expr;
   for(double coef = 1.0 / a.size(); const auto& [j,a] : a | views::enumerate) {
      expr += coef * gen_abs_expr(
         model, a,
         format("{}_{}",var_name,j),
         format("{}_N{}",constr_name,j),
         true,true
      );
   }
   return expr;
}

template<typename T>
GRBVar gen_l1a_var(
   GRBModel& model, const vec<T>& a,
   const string& var_name, const string& constr_name
) {
   return gen_var(
      model,gen_l1a_expr(model,a,var_name,constr_name),
      format("{}_output",var_name),
      format("{}_output",constr_name),
      0
   );
}

template<typename T>
GRBLinExpr gen_class_error_expr(
   GRBModel& model, const vec<T>& y, const string& var_name,
   const string& constr_name, const vec<double>& ty,
   double zero_tolerance = 0.0001, bool constraint = false,
   double constraint_tolerance = 0.1
) {
   vec<tuple<double,GRBLinExpr,int>> c_order;
   for(const auto& tup : views::zip( ty, y, views::iota(0) )) {
      c_order.emplace_back(tup);
   }
   ranges::sort( c_order, [](const auto& tuple_a, const auto& tuple_b) {
      return get<double>(tuple_a) < get<double>(tuple_b);
   } );
   const auto& pred = [zero_tolerance](const auto& tuple) {
      const auto& [ty,y,j] = tuple;
      return ty <= zero_tolerance;
   };
   GRBLinExpr expr;
   for(const auto& [ty,y,j] : c_order | views::take_while(pred)) {
      expr += y;
   }
   auto non_zero = c_order | views::drop_while(pred);
   for(const auto& [slide,tuple] : non_zero | views::enumerate) {
      const auto& [tc_a,y_a,j_a] = tuple;
      for(const auto&[tc_b,y_b,j_b] : non_zero | views::drop(slide + 1)) {
         GRBVar constrvio = model.addVar(
            -GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS,
            format("{}_{}_{}_constrvio", var_name, j_a, j_b)
         );
         model.addConstr(
            y_a - y_b + constrvio == log(tc_a / tc_b),
            format("{}_Rel{},{}", constr_name, j_a, j_b)
         );
         GRBLinExpr var_expr = gen_abs_expr(
            model, constrvio,
            format("{}_{}_{}_abs", var_name,    j_a, j_b),
            format("{}_Err{},{}",  constr_name, j_a, j_b),
            !constraint,!constraint
         );
         expr += var_expr;
         if(constraint) {
            model.addConstr(
               var_expr / log(tc_a / tc_b) <= constraint_tolerance,
               format("{}_RelConstr{},{}", constr_name, j_a, j_b)
            );
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
   double constraint_tolerance = 0.1
) {
   return gen_var(
      model,
      gen_class_error_expr(
         model,y,
         var_name,constr_name,
         ty,zero_tolerance,
         constraint,
         constraint_tolerance
      ),
      format("{}_output",var_name),
      format("{}_output",constr_name),
      0
   );
}

template<typename T>
GRBLinExpr gen_regression_error_expr(
   GRBModel& model, const vec<T>& y, const string& var_name,
   const string& constr_name, const vec<double>& ty,
   bool constraint = false, double constraint_tolerance = 0.1
) {
   GRBLinExpr expr;
   for(const auto& [i,y,ty] : views::zip( views::iota(0), y, ty )) {
      GRBLinExpr var_expr = gen_abs_error_expr(
         model, y, ty,
         format("{}_{}",  var_name,    i),
         format("{}_N{}", constr_name, i),
         !constraint,!constraint
      );
      expr += var_expr;
      if(constraint) {
         model.addConstr(
            var_expr / ty <= constraint_tolerance,
            format("{}_NConstr{}", constr_name,i)
         );
      }
   }
   return expr;
}

template<typename T>
GRBVar gen_regression_error_var(
   GRBModel& model, const vec<T>& y, const string& var_name,
   const string& constr_name, const vec<double>& ty,
   bool constraint = false, double constraint_tolerance = 0.1
) {
   return gen_var(
      model,
      gen_regression_error_expr(
         model,y,
         var_name,constr_name,
         ty,constraint,
         constraint_tolerance
      ),
      format("{}_output",var_name),
      format("{}_output",constr_name),
      0
   );
}

template<typename T>
vec<GRBLinExpr> gen_activation_exprs(
   GRBModel& model, const string& type, const vec<T>& z,
   const string& var_suffix, const string constr_suffix,
   const vec<double>& LeakyReLU_coef, const pair<double,double>& hardtanh_limits
) {
   vec<GRBLinExpr> a;
   if(type == "ReLU") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_ReLU_expr(
            model, z,
            format("relu{}_{}",  var_suffix,    j),
            format("relu{}_N{}", constr_suffix, j)
         ));
      }
   }
   else if(type == "ReLU6") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_ReLU6_expr(
            model, z,
            format("relu6{}_{}",  var_suffix,    j),
            format("relu6{}_N{}", constr_suffix, j)
         ));
      }
   }
   else if(type == "PReLU" || type == "LeakyReLU") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_LeakyReLU_expr(
            model, z,
            format("Lrelu{}_{}", var_suffix,   j),
            format("Lrelu{}_N{}",constr_suffix,j),
            LeakyReLU_coef[j]
         ));
      }
   }
   else if(type  == "Hardtanh") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_hardtanh_expr(
            model, z,
            format("ht{}_{}", var_suffix,   j),
            format("ht{}_N{}",constr_suffix,j),
            hardtanh_limits
         ));
      }
   }
   else if(type == "Hardsigmoid") {
      for(const auto& [j,z] : views::enumerate(z)) {
         a.emplace_back(gen_hardsigmoid_expr(
            model, z,
            format("hs{}_{}", var_suffix,   j),
            format("hs{}_N{}",constr_suffix,j)
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
   const vec<double>& LeakyReLU_coef, const pair<double,double>& hardtanh_limits
) {
   return gen_vars(
      model,
      gen_activation_exprs(
            model,type,z,
            var_name,constr_name,
            LeakyReLU_coef,hardtanh_limits
      ),
      format("{}_outputs",var_name),
      format("{}_outputs",constr_name)
   );
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
   double zero_tolerance = 0.0001, double constr_frac = 0.0,
   double constraint_tolerance = 0.1
) {
   GRBModel model(environment);
   ten4<GRBLinExpr> b(L);
   vec<GRBLinExpr> bias(L);
   GRBLinExpr L1_expr;
   for(int k = 0; k < L; ++k) {
      cout << "Processing layer: " << k << "\n";
      b[k] = ten3<GRBLinExpr>(C[k] + 1,mat<GRBLinExpr>(C[k + 1]));
      mat<GRBLinExpr> w(C[k] + 1,vec<GRBLinExpr>(C[k + 1]));
      for(int i = 0; i <= C[k]; ++i) {
         for(int j = 0; j < C[k + 1]; ++j) {
            b[k][i][j] = vec<GRBLinExpr>(D[k][i][j] + 1);
            for(int l = 0; l <= D[k][i][j]; ++l) {
               b[k][i][j][l] = gen_bin_expr(model, format("b_{}_{}_{}_{}",k,i,j,l), mask[k][i][j]);
            }
            w[i][j] = gen_w_var(
               model, b[k][i][j],
               format("w_{}_{}_{}",k,i,j), format("w_{}_{}_{}",k,i,j),
               precis[k][i][j]
            );
         }
      }
      if(l1w_norm[k]) {
         L1_expr += *l1w_norm[k] * gen_l1w_expr(model, w, format("l1w_{}",k), format("l1w_L{}",k));
      }
      bias[k] = model.addVar(bias_w[k], bias_w[k], 0, GRB_CONTINUOUS, format("bias_{}",k));
   }
   int constr_size = constr_frac * fx.size();
   GRBLinExpr EC_expr, ER_expr;
   for(const auto& [t,fx,tc_y,treg_y] : views::zip(views::iota(0),fx,class_ty,reg_ty)) {
      cout << "Processing case: " << t << "\n";
      vec<GRBLinExpr> a = gen_input_exprs(model, format("x_{}",t), fx, dropout.back());
      for(const auto& [k, b, AF, D, precis, bias, dropout, leakyReLU_coef, hardtanh_limits] : views::zip(
         views::iota(0), b, AF, D, precis, bias, views::drop(dropout,1), leakyReLU_coef, hardtanh_limits
      )) {
         cout << "Processing layer: " << k << "\n";
         const vec<GRBLinExpr>& z = gen_layer_exprs(
            model, a, b, bias,
            format("_{}_{}",t,k), format("_L{}_C{}",k,t),
            C[k], C[k + 1], precis,
            mask[k], dropout
         );
         a = gen_activation_exprs(
            model, AF, z,
            format("_{}_{}",t,k), format("_L{}_C{}",k,t),
            leakyReLU_coef, hardtanh_limits
         );
         if(l1a_norm[k]) {
            L1_expr += *l1a_norm[k] * gen_l1a_expr(
               model, a,
               format("l1a_{}_{}",t,k), format("l1a_L{}_C{}",k,t)
            );
         }
      }
      const auto& ry_view = a | views::take(treg_y.size());
      const auto& cy_view = a | views::drop(treg_y.size());
      if(t <= constr_size) {
         gen_class_error_expr(
            model, vec<GRBLinExpr>( cy_view.begin(), cy_view.end() ),
            format("EC_{}",t), format("ClassE_{}",t),
            tc_y, zero_tolerance,true,constraint_tolerance
         );
         gen_regression_error_expr(
            model, vec<GRBLinExpr>( ry_view.begin(), ry_view.end() ),
            format("ER_{}",t), format("RegE_{}",t),
            treg_y,true,constraint_tolerance
         );
      }
      else {
         EC_expr += gen_class_error_expr(
            model, vec<GRBLinExpr>( cy_view.begin(), cy_view.end() ),
            format("EC_{}",t), format("ClassE_{}",t),
            tc_y, zero_tolerance
         );
         ER_expr += gen_regression_error_expr(
            model, vec<GRBLinExpr>( ry_view.begin(), ry_view.end() ),
            format("ER_{}",t), format("RegE_{}",t),
            treg_y
         );
      }
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
      {"zero_tolerance", 1},
      {"constr_tol", 1},
      {"constr_frac", 1},
      {"mst_file", 1}
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
   auto regression_targets = read_matrix_from_csv<double>(fstream(regression_targets_path));
   auto class_targets = read_matrix_from_csv<double>(fstream(class_targets_path));
   int T = features.size(), L = C.size() - 1;
   if(class_targets.empty()) {
      class_targets = mat<double>(T);
   }
   if(regression_targets.empty()) {
      regression_targets = mat<double>(T);
   }
   vec<mat<int>> bits;
   if(opts.contains("use_bits")) {
      path file_path = load_path / format("{}.csv",safe_suffix(load_name,"bits"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() == 2) {
         bits = get_layers_matrix(dim,data);
      }
   }
   if(bits.empty()) {
      bits = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         bits[k] = mat<int>(C[k] + 1,vec<int>(C[k + 1]));
         for(auto& row : bits[k]) {
            for(auto& value : row) {
               value = 4;
            }
         }
      }
   }
   vec<mat<int>> precision;
   if(opts.contains("use_precision")) {
      path file_path = load_path / format("{}.csv",safe_suffix(load_name,"exp"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() == 2) {
         precision = get_layers_matrix(dim,data);
      }
   }
   if(precision.empty()) {
      precision = vec<mat<int>>(L);
      for(int k = 0; k < L; ++k) {
         precision[k] = mat<int>(C[k] + 1,vec<int>(C[k + 1]));
         for(auto& row : precision[k]) {
            for(auto& value : row) {
               value = 2;
            }
         }
      }
   }
   vec<mat<int>> mask;
   if(opts.contains("use_mask")) {
      path file_path = load_path / format("{}.csv",safe_suffix(load_name,"mask"));
      const auto& [dim, data] = read_list_from_csv<int>(fstream(file_path));
      if(dim.size() == 2) {
         mask = get_layers_matrix(dim,data);
      }
   }
   if(mask.empty()) {
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
   string ResultFile = format("{}.sol.xz",file_path);
   string SolFiles = file_path;
   string LogFile = format("{}.log.xz",file_path);
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
      argstream = stringstream(args[0]);
      argstream >> BestObjStop;
      ambiente.set(GRB_DoubleParam_BestObjStop, BestObjStop);
   });
   double zero_tolerance = 0.001;
   process_yes_arg(opts, "zero_tolerance", [&argstream,&zero_tolerance](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> zero_tolerance;
   });
   double constraint_tolerance = 0.1;
   process_yes_arg(opts, "constr_tol", [&argstream,&constraint_tolerance](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> constraint_tolerance;
   });
   double constraint_frac = 0.0;
   process_yes_arg(opts, "constr_tol", [&argstream,&constraint_frac](const auto& args) {
      argstream = stringstream(args[0]);
      argstream >> constraint_frac;
   });
   process_yes_arg(opts, "feas_tol", [&argstream,&ambiente](const auto& args) {
      double FeasibilityTol;
      argstream = stringstream(args[0]);
      argstream >> FeasibilityTol;
      ambiente.set(GRB_DoubleParam_FeasibilityTol, FeasibilityTol);
   });
   process_yes_arg(opts, "int_feas_tol", [&argstream,&ambiente](const auto& args) {
      double IntFeasTol;
      argstream = stringstream(args[0]);
      argstream >> IntFeasTol;
      ambiente.set(GRB_DoubleParam_IntFeasTol, IntFeasTol);
   });
   process_yes_arg(opts, "iteration_limit", [&argstream,&ambiente](const auto& args) {
      double IterationLimit;
      argstream = stringstream(args[0]);
      argstream >> IterationLimit;
      ambiente.set(GRB_DoubleParam_IterationLimit, IterationLimit);
   });
   process_yes_arg(opts, "opt_tol", [&argstream,&ambiente](const auto& args) {
      double OptimalityTol;
      argstream = stringstream(args[0]);
      argstream >> OptimalityTol;
      ambiente.set(GRB_DoubleParam_OptimalityTol, OptimalityTol);
   });
   process_yes_arg(opts, "solution_limit", [&argstream,&ambiente](const auto& args) {
      int SolutionLimit;
      argstream = stringstream(args[0]);
      argstream >> SolutionLimit;
      ambiente.set(GRB_IntParam_SolutionLimit, SolutionLimit);
   });
   process_yes_arg(opts, "time_limit", [&argstream,&ambiente](const auto& args) {
      double TimeLimit;
      argstream = stringstream(args[0]);
      argstream >> TimeLimit;
      ambiente.set(GRB_DoubleParam_TimeLimit, TimeLimit);
   });
   process_yes_arg(opts, "node_limit", [&argstream,&ambiente](const auto& args) {
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
   GRBModel modelo = get_model(
      ambiente, T, L, C, AF,
      mask, precision, bits, bias,
      leakyReLU, hardtanh, dropout, l1a_norm, l1w_norm,
      features, class_targets, regression_targets,
      zero_tolerance, constraint_frac, constraint_tolerance
   );
   if(save_lp)
      modelo.write(path(format("{}.lp.xz",file_path)).string());
   if(optimize) {
      process_yes_arg(opts, "mst_file", [&argstream,&modelo](const auto& args) {
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
               modelo.write(format("{}.sol.xz",file_path));
            }
            if(save_json) {
               modelo.write(format("{}.json.xz",file_path));
            } 
            if(save_mst) {
               modelo.write(format("{}.mst.xz",file_path));
            }
            break;
         case GRB_INFEASIBLE :
            cout << "Modelo infactible\n";
            if(save_ilp) {
               modelo.computeIIS( );
               modelo.write(format("{}.ilp.xz",file_path));
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
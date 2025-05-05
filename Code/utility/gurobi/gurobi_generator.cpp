#include <format>
#include <iostream>
#include <gurobi_c++.h>
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

namespace fsys = std::filesystem;
using std::fstream;
using fsys::path;
using std::format;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::optional;
using std::pair;
using std::getline;
using std::stoi;
using std::stod;
using std::cout;

template<typename T>
using vec = std::vector<T>;
template<typename T>
using mat = vec<vec<T>>;
template<typename T>
using ten3 = vec<vec<vec<T>>>;
template<typename T>
using ten4 = vec<vec<vec<vec<T>>>>;
template<typename T>
using ten5 = vec<vec<vec<vec<vec<T>>>>>;

auto layer_binary_variables(GRBModel& model, int L, const vec<int>& C, const ten3<int>& D, const optional<ten3<bool>>& mask = {}) {
   ten4<GRBVar> b(L);
   for(int k = 0; k < L; ++k) {
      b[k] = ten3<GRBVar>(C[k] + 1);
      for(int i = 0; i <= C[k]; ++i) {
         b[k][i] = mat<GRBVar>(C[k + 1]);
         for(int j = 0; j < C[k + 1]; ++j)
            if(mask && (*mask)[k][i][j] || !mask) {
               b[k][i][j] = vec<GRBVar>(D[k][i][j] + 1);
               for(int l = 0; l <= D[k][i][j]; ++l)
                  b[k][i][j][l] = model.addVar(0, 1, 0, GRB_BINARY, format("b_{}_{}_{}_{}",  k, i, j, l));
            }
      }
   }
   return b;
}

auto pounded_activation_variables(GRBModel& model, int T, int L, const vec<int>& C, const ten3<int>& D, const optional<ten3<bool>>& mask = {}) {
   ten5<GRBVar> bw(T,ten4<GRBVar>(L));
   for(int t = 0; t < T; ++t)
      for(int k = 0; k < L; ++k) {
         bw[t][k] = ten3<GRBVar>(C[k] + 1);
         for(int i = 0; i <= C[k]; ++i) {
            bw[t][k][i] = mat<GRBVar>(C[k + 1]);
            for(int j = 0; j < C[k + 1]; ++j)
               if(mask && (*mask)[k][i][j] || !mask) {
                  bw[t][k][i][j] = vec<GRBVar>(D[k][i][j] + 1);
                  for(int l = 0; l <= D[k][i][j]; ++l)
                     bw[t][k][i][j][l] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("bw_{}_{}_{}_{}_{}", t, k, i, j, l));
               }
         }
      }
   return bw;
}

auto linear_activation_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   ten3<GRBVar> z(T,mat<GRBVar>(L));
   for(int t = 0; t < T; ++t)
      for(int k = 0; k < L; ++k) {
         z[t][k] = vec<GRBVar>(C[k + 1]);
         for(int j = 0; j < C[k + 1]; ++j)
            z[t][k][j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("z_{}_{}_{}", t, k, j));
      }
   return z;
}

auto total_error_variables(GRBModel& model, int T) {
   vec<GRBVar> E(T); 
   for(int t = 0; t < T; ++t)
      E[t] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("E_{}", t));
   return E;
}

auto output_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   mat<GRBVar> y(T,vec<GRBVar>(C[L]));
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[L]; ++j)
         y[t][j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("y_{}_{}", t, j));  
   return y;
}

auto abs_error_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   mat<GRBVar> dy(T,vec<GRBVar>(C[L]));
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[L]; ++j)
         dy[t][j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("dy_{}_{}", t, j));
   return dy;
}

auto error_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   mat<GRBVar> e(T,vec<GRBVar>(C[L]));
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[L]; ++j)
         e[t][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("e_{}_{}", t, j));
   return e;
}

auto input_variables(GRBModel& model, int T, const vec<int>& C, const mat<double>& px) {
   mat<GRBVar> x(T,vec<GRBVar>(C[0]));
   for(int t = 0; t < T; ++t)
      for(int i = 0; i < C[0]; ++i)
         x[t][i] = model.addVar(px[t][i], px[t][i], 0, GRB_CONTINUOUS, format("x_{}_{}", t, i));
   return x;
}

auto activation_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   ten3<GRBVar> a(T,mat<GRBVar>(L + 1));
   for(int t = 0; t < T; ++t)
      for(int k = 0; k <= L; ++k) {
         a[t][k] = vec<GRBVar>(C[k] + 1);
         for(int i = 0; i <= C[k]; ++i)
            a[t][k][i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("a_{}_{}_{}", t, k, i));
      }
   return a;
}

auto bias_variables(GRBModel& model, int L, const optional<vec<double>>& used_bias) {
   vec<GRBVar> bias(L);
   if (used_bias)
      for(int k = 0; k < L; ++k)
         bias[k] = model.addVar((*used_bias)[k], (*used_bias)[k], 0, GRB_CONTINUOUS, format("bias_{}", k));
   else
      for(int k = 0; k < L; ++k)
         bias[k] = model.addVar(1, 1, 0, GRB_CONTINUOUS, format("bias_{}", k));
   return bias;
}

auto auxiliary_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   ten4<GRBVar> aux(T,ten3<GRBVar>(L));
   for(int t = 0; t < T; ++t)
      for(int k = 0; k < L; ++k) {
         aux[t][k] = mat<GRBVar>(C[k + 1]);
         for(int j = 0; j < C[k + 1]; ++j) {
            aux[t][k][j] = vec<GRBVar>(2);
            aux[t][k][j][0] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("aux_{}_{}_{}_A", t, k, j));
            aux[t][k][j][1] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("aux_{}_{}_{}_B", t, k, j));
         }
      }
   return aux;
}

void absolute_error_constraints(GRBModel& model, const mat<GRBVar>& e, const mat<GRBVar>& y, const mat<GRBVar>& dy, int T, int L, const vec<int>& C, const mat<double>& ty) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[L]; ++j) {
         model.addConstr(y[t][j] - dy[t][j], GRB_EQUAL, ty[t][j], format("Diff_Pred_N{}_C{}", j, t));
         model.addGenConstrAbs(e[t][j], dy[t][j], format("Abs_e_N{}_C{}", j, t));
      }
}

void sum_error_constraints(GRBModel& model, const GRBVar& J, const vec<GRBVar>& E, const mat<GRBVar>& e, int T, int L, const vec<int>& C) {
   GRBLinExpr case_sum_error_expr;
   for(int t = 0; t < T; ++t) {
      GRBLinExpr sum_error_expr;
      for(int j = 0; j < C[L]; ++j)
         sum_error_expr += e[t][j];
      model.addConstr(sum_error_expr, GRB_EQUAL, E[t], format("Abs_E_C{}", t));
      case_sum_error_expr += E[t];
   }
   model.addConstr(case_sum_error_expr, GRB_EQUAL, J, format("Abs_T_E"));
}

void max_error_objective(GRBModel& model, const GRBVar& J, const vec<GRBVar>& E, const mat<GRBVar>& e, int T, int L, const vec<int>& C) {
   GRBLinExpr case_sum_error_expr;
   for(int t = 0; t < T; ++t) {
      model.addGenConstrMax(E[t], e[t].data(), e[t].size(), 0, format("Max_Abs_E_C{}", t));
      case_sum_error_expr += E[t];
   }
   model.addConstr(case_sum_error_expr, GRB_EQUAL, J,format("Max_Abs_T_E"));
}

void max_total_error_objective(GRBModel& model, const GRBVar& J, const vec<GRBVar>& E, const mat<GRBVar>& e, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      model.addGenConstrMax(E[t], e[t].data(), e[t].size(), 0, format("Max_Abs_E_C{}", t));
   model.addGenConstrMax(J, E.data(), E.size(), 0, format("Max_T_Abs_E"));
}

void output_layer_constraints(GRBModel& model, const ten3<GRBVar>& a, const mat<GRBVar>& y, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[L]; ++j)
         model.addConstr(y[t][j], GRB_EQUAL, a[t][L][j], format("Output_N{}_C{}", j, t));
}

void hardtanh_constraints(GRBModel& model, const ten3<GRBVar>& z, const ten4<GRBVar>& aux, const ten3<GRBVar>& a, int T, int k, const vec<int>& C, const pair<double,double>& limits = {-1,1}) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMin(aux[t][k][j][0], &z[t][k][j], 1, limits.first, format("L{}_N{}_Hardtanh_A_C{}", k, j, t));
         model.addGenConstrMax(a[t][k + 1][j], &aux[t][k][j][0], 1, limits.second, format("L{}_N{}_Hardtanh_B_C{}", k, j, t));
      }
}

void hardsigmoid_constraints(GRBModel& model, const ten3<GRBVar>& z, const ten4<GRBVar>& aux, const ten3<GRBVar>& a, int T, int k, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addConstr(z[t][k][j] / 6 + 0.5, GRB_EQUAL, aux[t][k][j][0], format("L{}_N{}_Hardsigmoid_A_C{}", k, j, t));
         model.addGenConstrMin(aux[t][k][j][1],&aux[t][k][j][0],1,1, format("L{}_N{}_Hardsigmoid_B_C{}", k, j, t));
         model.addGenConstrMax(a[t][k + 1][j],&aux[t][k][j][1],1,0, format("L{}_N{}_Hardsigmoid_C_C{}", k, j, t));
      }
}

void ReLU6_constraints(GRBModel& model, const ten3<GRBVar>& z, const ten4<GRBVar>& aux, const ten3<GRBVar>& a, int T, int k, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, format("L{}_N{}_RELU6_A_C{}", k, j, t));
         model.addGenConstrMin(a[t][k + 1][j],&aux[t][k][j][0],1,6, format("L{}_N{}_RELU6_B_C{}", k, j, t));
      }
}

void ReLU_constraints(GRBModel& model, const ten3<GRBVar>& z, const ten4<GRBVar>& aux, const ten3<GRBVar>& a, int T, int k, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[k + 1]; ++j)
         model.addGenConstrMax(a[t][k + 1][j],&z[t][k][j],1,0, format("L{}_N{}_ReLU_C{}", k + 1, j, t));
}

void direct_pass_constraints(GRBModel& model, const ten3<GRBVar>& z, const ten4<GRBVar>& aux, const ten3<GRBVar>& a, int T, int k, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[k + 1]; ++j)
         model.addConstr(a[t][k + 1][j], GRB_EQUAL, z[t][k][j], format("L{}_N{}_ReLU_C{}", k + 1, j, t));
}

void LeakyReLU_constraints(GRBModel& model, const ten3<GRBVar>& z, const ten4<GRBVar>& aux, const ten3<GRBVar>& a, int T, int k, const vec<int>& C, double neg_coef = 0.25) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, format("L{}_N{}_LeakyReLU_A_C{}", k, j, t));
         model.addGenConstrMin(aux[t][k][j][1],&z[t][k][j],1,0, format("L{}_N{}_LeakyReLU_B_C{}", k, j, t));
         model.addConstr(a[t][k + 1][j] == aux[t][k][j][0] + neg_coef * aux[t][k][j][1], format("L{}_N{}_LeakyReLU_C_C{}", k, j, t));
      }
}

void PReLU_constraints(GRBModel& model, const ten3<GRBVar>& z, const ten4<GRBVar>& aux, const ten3<GRBVar>& a, int T, int k, const vec<int>& C, const optional<vec<double>>& neg_coef = {}) {
   for(int t = 0; t < T; ++t)
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, format("L{}_N{}_PReLU_A_C{}", k, j, t));
         model.addGenConstrMin(aux[t][k][j][1],&z[t][k][j],1,0, format("L{}_N{}_PReLU_B_C{}", k, j, t));
         if(neg_coef && (*neg_coef).size() > 0)
            model.addConstr(a[t][k + 1][j] == aux[t][k][j][0] + (*neg_coef)[j] * aux[t][k][j][1], format("L{}_N{}_PReLU_C_C{}", k, j, t));
         else
            model.addConstr(a[t][k + 1][j] == aux[t][k][j][0] + 0.25 * aux[t][k][j][1], format("L{}_N{}_PReLU_C_C{}", k, j, t));
      }
}

void bias_constraints(GRBModel& model, const vec<GRBVar>& layer_bias, const ten3<GRBVar>& a, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      for(int k = 0; k < L; ++k)
         model.addConstr(layer_bias[k], GRB_EQUAL, a[t][k][C[k]], format("L{}_bias_C{}",k,t));
}

void binary_constraints(GRBModel& model, const ten4<GRBVar>& b, const ten5<GRBVar>& bw, const ten3<GRBVar>& a, const ten3<GRBVar>& z, int T, int L, const vec<int>& C, const ten3<int>& D, const optional<ten3<int>>& precis = {}, const optional<ten3<bool>>& mask = {}) {
   for(int t = 0; t < T; ++t)
      for(int k = 0; k < L; ++k)
         for(int i = 0; i <= C[k]; ++i)
            for(int j = 0; j < C[k + 1]; ++j)
               if(mask && (*mask)[k][i][j] || !mask)
                  for(int l = 0; l <= D[k][i][j]; ++l) {
                     if(precis)
                        model.addGenConstrIndicator(b[k][i][j][l], 1, exp2(l - (*precis)[k][i][j]) * a[t][k][i] - bw[t][k][i][j][l], GRB_EQUAL, 0, format("L{}_W{},{}_D{}_Descomp_A_C{}",k,i,j,l,t));
                     else
                        model.addGenConstrIndicator(b[k][i][j][l], 1, exp2(l) * a[t][k][i] - bw[t][k][i][j][l], GRB_EQUAL, 0, format("L{}_W{},{}_D{}_Descomp_A_C{}",k,i,j,l,t));
                     model.addGenConstrIndicator(b[k][i][j][l], 0, bw[t][k][i][j][l], GRB_EQUAL, 0, format("L{}_W{},{}_D{}_Descomp_B_C{}",k,i,j,l,t));
                  }
   for(int t = 0; t < T; ++t)
      for(int k = 0; k < L; ++k)
         for(int j = 0; j < C[k + 1]; ++j) {
            GRBLinExpr neuron_linear_expr;
            for(int i = 0; i <= C[k]; ++i)
               if(mask && (*mask)[k][i][j] || !mask) {
                  for(int l = 0; l < D[k][i][j]; ++l) {
                     neuron_linear_expr += bw[t][k][i][j][l];
                  }
                  neuron_linear_expr -= bw[t][k][i][j][D[k][i][j]];
               }
            model.addConstr(neuron_linear_expr, GRB_EQUAL, z[t][k][j], format("Lin_Output_L{}_N{}_C{}",k,j,t));
         }
}

void input_layer_constraints(GRBModel& model, const ten3<GRBVar>& a, const mat<GRBVar>& x, int T, const vec<int>& C) {
   for(int t = 0; t < T; ++t)
      for(int i = 0; i < C[0]; ++i)
         model.addConstr(a[t][0][i], GRB_EQUAL, x[t][i], format("Input_N{}_C{}", t, i));
}

enum Obj_Func {
   SUM,
   MAX_e,
   MAX_E
};

GRBModel get_model(const GRBEnv& environment, int T, int L, const vec<int>& C, const vec<string>& AF, const ten3<int>& D, const mat<double>& tx, const mat<double>& ty, Obj_Func optim_opt = Obj_Func::SUM, const optional<ten3<bool>>& mask = {}, const optional<ten3<int>>& precis = {}, const optional<vec<double>>& bias_w = {}, const optional<vec<double>>& LeakyReLU_coef = {}, const optional<mat<double>>& PReLU_coef = {}, const optional<vec<pair<double,double>>>& hardtanh_limits = {}) {
   GRBModel model(environment);
   GRBVar J = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("J"));
   auto E = total_error_variables(model,T);
   auto e = error_variables(model,T,L,C);
   auto dy = abs_error_variables(model,T,L,C);
   auto y = output_variables(model,T,L,C);
   auto aux = auxiliary_variables(model,T,L,C);
   auto z = linear_activation_variables(model,T,L,C);
   auto bw = pounded_activation_variables(model,T,L,C,D,mask);
   auto b = layer_binary_variables(model,L,C,D,mask);
   auto bias = bias_variables(model,L,bias_w);
   auto a = activation_variables(model,T,L,C);
   auto x = input_variables(model,T,C,tx);
   
   model.setObjective(GRBLinExpr(J), GRB_MINIMIZE);
   switch(optim_opt) {
      case Obj_Func::MAX_E :
         max_total_error_objective(model,J,E,e,T,L,C);
         break;
      case Obj_Func::MAX_e :
         max_error_objective(model,J,E,e,T,L,C);
         break;
      case Obj_Func::SUM :
         sum_error_constraints(model,J,E,e,T,L,C);
   }
   absolute_error_constraints(model,e,y,dy,T,L,C,ty);
   output_layer_constraints(model,a,y,T,L,C);
   for(int k = 0; k < L; ++k)
      if(AF[k] == "ReLU")
         ReLU_constraints(model,z,aux,a,T,k,C);
      else if(AF[k] == "ReLU6")
         ReLU6_constraints(model,z,aux,a,T,k,C);
      else if(AF[k] == "PReLU")
         if(PReLU_coef && (*PReLU_coef).size() > k  && (*PReLU_coef)[k].size() > 0)
            PReLU_constraints(model,z,aux,a,T,k,C,(*PReLU_coef)[k]);
         else
            PReLU_constraints(model,z,aux,a,T,k,C);
      else if(AF[k] == "LeakyReLU")
         if(LeakyReLU_coef && (*LeakyReLU_coef).size() > k)
            LeakyReLU_constraints(model,z,aux,a,T,k,C,(*LeakyReLU_coef)[k]);
         else
            LeakyReLU_constraints(model,z,aux,a,T,k,C);
      else if(AF[k]  == "Hardtanh")
         if(hardtanh_limits && (*hardtanh_limits).size() > k)
            hardtanh_constraints(model,z,aux,a,T,k,C,(*hardtanh_limits)[k]);
         else
            hardtanh_constraints(model,z,aux,a,T,k,C);
      else if(AF[k] == "Hardsigmoid")
         hardsigmoid_constraints(model,z,aux,a,T,k,C);
      else
         direct_pass_constraints(model,z,aux,a,T,k,C);
   bias_constraints(model,bias,a,T,L,C);
   binary_constraints(model,b,bw,a,z,T,L,C,D,precis);
   input_layer_constraints(model,a,x,T,C);
   return model;
}

template<typename T>
auto read_matrix(const path& matrix_path) {
   fstream matrix_stream(matrix_path);
   string line, word;
   mat<T> matrix;
   getline(matrix_stream, line);
   while(getline(matrix_stream, line)) {
      replace(line.begin(),line.end(),',',' ');
      stringstream line_stream(line);
      vec<T> row;
      line_stream >> word;
      T value;
      while(line_stream  >> std::noboolalpha >> value) {
         row.push_back(value);
      }
      matrix.push_back(row);
   }
   return matrix;
}

pair<vec<int>,vec<string>> read_arch(const path& arch_path) {
   fstream arch_stream(arch_path);
   string line, word;
   vec<int> C;
   vec<string> AF;
   getline(arch_stream, line);
   while(getline(arch_stream, line)) {
      stringstream line_stream(line);
      getline(line_stream, word, ',');
      getline(line_stream, word, ',');
      C.push_back(stoi(word));
      getline(line_stream, word, ',');
      AF.push_back(word);
   }
   return {C, AF};
}

auto read_hardtanh(const path& list_path) {
   fstream list_stream(list_path);
   string line, word;
   vec<pair<double,double>> hardtanh_params;
   getline(list_stream, line);
   while(getline(list_stream, line)) {
      replace(line.begin(),line.end(),',',' ');
      stringstream line_stream(line);
      double min = -1, max = 1;
      line_stream >> word >> min >> max;
      hardtanh_params.push_back({min,max});
   }
   return hardtanh_params;
}

template<typename T>
auto read_matrix_list(const path& list_path) {
   fstream list_stream(list_path);
   string line, word;
   ten3<T> list;
   getline(list_stream, line);
   while(getline(list_stream, line)) {
      replace(line.begin(),line.end(),',',' ');
      stringstream line_stream(line);
      int n, m;
      line_stream  >> std::noboolalpha >> word >> n >> m;
      mat<T> matrix(n,vec<T>(m));
      for(int i = 0; i < n; ++i) {
         for(int j = 0; j < m; ++j) {
            T value;
            line_stream >> std::noboolalpha >> value;
            matrix[i][j] = value;
         }
      }
      list.push_back(matrix);
   }
   return list;
}

template<typename T>
auto read_vector_list(const path& list_path) {
   fstream list_stream(list_path);
   string line, word;
   mat<T> list;
   getline(list_stream, line);
   while(getline(list_stream, line)) {
      replace(line.begin(),line.end(),',',' ');
      stringstream line_stream(line);
      int n;
      line_stream >> std::noboolalpha >> word >> n;
      vec<T> vector(n);
      for(int i = 0; i < n; ++i) {
         T value;
         line_stream >> std::noboolalpha >> value;
         vector[i] = value;
      }
      list.push_back(vector);
   }
   return list;
}

template<typename T>
auto read_vector(const path& vector_path) {
   fstream vector_stream(vector_path);
   string line, word;
   vec<T> vector;
   getline(vector_stream, line);
   while(getline(vector_stream, line)) {
      replace(line.begin(), line.end(),',',' ');
      stringstream line_stream(line);
      T value;
      line_stream >> std::noboolalpha >> word >> value;
      vector.push_back(value);
   }
   return vector;
}

optional<unordered_map<string,pair<optional<vec<string>>,int>>> process_opts(int argc, const char* argv[]) {
   unordered_map<string,pair<optional<vec<string>>,int>> opts = {
      {"load_path", {{},1}},
      {"load_name", {{},1}},
      {"save_path", {{},1}},
      {"save_name", {{},1}},
      {"time_limit", {{},1}},
      {"solution_limit", {{},1}},
      {"iteration_limit", {{},1}},
      {"node_limit", {{},1}},
      {"opt_tol", {{},1}},
      {"best_obj_stop", {{},1}},
      {"feas_tol", {{},1}},
      {"int_feas_tol", {{},1}},
      {"no_log_to_console", {{},0}},
      {"no_save_sols", {{},0}},
      {"no_save_log", {{},0}},
      {"no_save_json", {{},0}},
      {"no_save_sol", {{},0}},
      {"no_save_mst", {{},0}},
      {"no_save_ilp", {{},0}},
      {"no_save_lp", {{},0}},
      {"no_optimize", {{},0}},
      {"obj_type", {{},1}},
      {"use_precision", {{},0}},
      {"use_bias", {{},0}},
      {"use_mask", {{},0}},
      {"use_PReLU", {{},0}},
      {"use_leakyReLU", {{},0}},
      {"use_hardtanh", {{},0}}
   };
   int argi = 1;
   string arg, arg_name;
   while(argi < argc) {
      arg = argv[argi];
      if(arg.starts_with("--")) {
         arg_name = arg.substr(2);
         if(opts.contains(arg_name)) {
            int i = 1;
            for(opts[arg_name].first = vec<string>(); i < opts[arg_name].second && argi + i < argc; ++i)
               (*opts[arg_name].first).push_back(argv[argi + i]);
            argi += i;
         }
         else
            return {};
      }
   }
   return optional(opts);
}

int main(int argc, const char* argv[]) try {
   // lectura de la entrada
   auto opts = process_opts(argc,argv);
   if(!opts) {
      cout << "Argumentos invalidos\n";
      return 0;
   }
   path save_path("");
   path load_path("");
   string save_name("model");
   string load_name("");
   auto opt = (*opts)["load_path"].first;
   if(opt)
      load_path = path((*opt)[0]);
   opt = (*opts)["save_path"].first;
   if(opt)
      save_path = path((*opt)[0]);
   opt = (*opts)["save_name"].first;
   if(opt)
      save_name = format("{}_", (*opt)[0]);
   opt = (*opts)["load_name"].first;
   if(opt)
      load_name = format("{}_", (*opt)[0]);
      
   path arch_path = load_path / format("{}arch.csv",load_name);
   path cases_path = load_path / format("{}cases.csv",load_name);
   path labels_path = load_path / format("{}labels.csv",load_name);
   path digits_path = load_path / format("{}digits.csv",load_name);

   auto [C, AF] = read_arch(arch_path);
   auto cases = read_matrix<double>(cases_path);
   auto labels = read_matrix<double>(labels_path);
   auto digits = read_matrix_list<int>(digits_path);
   int T = cases.size();
   int L = C.size();

   optional<ten3<int>> precision;
   opt = (*opts)["use_precision"].first;
   if(opt) {
      precision = optional(read_matrix_list<int>(load_path / format("{}precision.csv",load_name)));
   }
   optional<vec<double>> bias;
   opt = (*opts)["use_bias"].first;
   if(opt) {
      bias = optional(read_vector<double>(load_path / format("{}bias.csv",load_name)));
   }
   optional<ten3<bool>> mask;
   opt = (*opts)["use_mask"].first;
   if(opt) {
      mask = optional(read_matrix_list<bool>(load_path / format("{}mask.csv",load_name)));
   }
   optional<vec<double>> leakyReLU;
   opt = (*opts)["use_leakyReLU"].first;
   if(opt) {
      leakyReLU = optional(read_vector<double>(load_path / format("{}LeakyReLU.csv",load_name)));
   }
   optional<mat<double>> PReLU;
   opt = (*opts)["use_PReLU"].first;
   if(opt) {
      PReLU = optional(read_vector_list<double>(load_path / format("{}PReLU.csv",load_name)));
   }
   optional<vec<pair<double,double>>> hardtanh;
   opt = (*opts)["use_hardtanh"].first;
   if(opt) {
      hardtanh = optional(read_hardtanh(load_path / format("{}hardtanh.csv",load_name)));
   }
   
   string file_path = save_path / save_name;
   string ResultFile = format("{}.sol",file_path);
   GRBEnv ambiente;
   
   opt = (*opts)["no_save_sols"].first;
   if(!opt) {
      string SolFiles = file_path;   
      ambiente.set(GRB_StringParam_SolFiles, SolFiles);
   }
   opt = (*opts)["no_save_log"].first;
   if(!opt) {
      string LogFile = format("{}.log",file_path);
      ambiente.set(GRB_StringParam_LogFile, LogFile);
   }
   int LogToConsole = (*opts)["no_log_to_console"].first.has_value();
   ambiente.set(GRB_IntParam_LogToConsole, LogToConsole);
   opt = (*opts)["best_obj_stop"].first;
   if(opt) {
      double BestObjStop = stod((*opt)[0]);
      ambiente.set(GRB_DoubleParam_BestObjStop, BestObjStop);
   }
   opt = (*opts)["feas_tol"].first;
   if(opt) {
      double FeasibilityTol = stod((*opt)[0]);
      ambiente.set(GRB_DoubleParam_FeasibilityTol, FeasibilityTol);
   }
   opt = (*opts)["int_feas_tol"].first;
   if(opt) {
      double IntFeasTol = stod((*opt)[0]);
      ambiente.set(GRB_DoubleParam_IntFeasTol, IntFeasTol); 
   }
   opt = (*opts)["iteration_limit"].first;
   if(opt) {
      double IterationLimit = stod((*opt)[0]);
      ambiente.set(GRB_DoubleParam_IterationLimit, IterationLimit); 
   }
   opt = (*opts)["opt_tol"].first;
   if(opt) {
      double OptimalityTol = stod((*opt)[0]);
      ambiente.set(GRB_DoubleParam_OptimalityTol, OptimalityTol);   
   }
   opt = (*opts)["solution_limit"].first;
   if(opt) {
      int SolutionLimit = stoi((*opt)[0]);
      ambiente.set(GRB_IntParam_SolutionLimit, SolutionLimit);    
   }
   opt = (*opts)["time_limit"].first;
   if(opt) {
      double TimeLimit = stod((*opt)[0]);
      ambiente.set(GRB_DoubleParam_TimeLimit, TimeLimit);     
   }
   opt = (*opts)["node_limit"].first;
   if(opt) {
      double NodeLimit = stod((*opt)[0]);
      ambiente.set(GRB_DoubleParam_NodeLimit, NodeLimit);     
   }
   int JSONSolDetail = 1;
   ambiente.set(GRB_IntParam_JSONSolDetail, JSONSolDetail);
   bool optimize = !(*opts)["no_optimize"].first;
   bool save_lp = !(*opts)["no_save_lp"].first;
   bool save_ilp = !(*opts)["no_save_ilp"].first; 
   bool save_sol = !(*opts)["no_save_sol"].first;
   bool save_mst = !(*opts)["no_save_mst"].first;
   bool save_json = !(*opts)["no_save_json"].first; 

   opt = (*opts)["obj_type"].first;
   Obj_Func error_type = Obj_Func::SUM;
   if(opt)
      if((*opt)[0] == "SUM")
         error_type = Obj_Func::SUM;
      else if((*opt)[0] == "MAX_e")
         error_type = Obj_Func::MAX_e;
      else if((*opt)[0] == "MAX_E")
         error_type = Obj_Func::MAX_E;
      else {
         std::cout << format("Error en el argumento {}\n", (*opt)[0]);
         return 0;
      }

   // construcción del modelo
   GRBModel modelo = get_model(ambiente,T,L,C,AF,digits,cases,labels,error_type,mask,precision,bias,leakyReLU,PReLU,hardtanh);
   // ------ resolución del modelo
   if(save_lp)
      modelo.write(path(format("{}.lp",file_path)));
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
            if(save_sol)
               modelo.write(format("{}.sol",file_path));
            if(save_json)
               modelo.write(format("{}.json",file_path));
            if(save_mst)
               modelo.write(format("{}.mst",file_path));
            break;
         case GRB_INFEASIBLE :
            std::cout << "Modelo infactible\n";
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
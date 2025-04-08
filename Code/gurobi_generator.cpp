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
#include <utility>

namespace fsys = std::filesystem;
using std::fstream;
using fsys::path;
using std::format;
using std::string;
using std::stringstream;

template<typename T>
using vec = std::vector<T>;

auto layer_binary_variables(GRBModel& model, int L, const vec<int>& C, const vec<vec<vec<int>>>& D) {
   vec<vec<vec<vec<GRBVar>>>> b(L);
   for(int k = 0; k < L; ++k) {
      b[k] = vec<vec<vec<GRBVar>>>(C[k] + 1);
      for(int i = 0; i <= C[k]; ++i) {
         b[k][i] = vec<vec<GRBVar>>(C[k + 1]);
         for(int j = 0; j < C[k + 1]; ++j) {
            b[k][i][j] = vec<GRBVar>(D[k][i][j] + 1);
            for(int l = 0; l <= D[k][i][j]; ++l) {
               b[k][i][j][l] = model.addVar(0, 1, 0, GRB_BINARY, format("b_{}_{}_{}_{}",  k, i, j, l));
            }
         }
      }
   }
   return b;
}

auto pounded_activation_variables(GRBModel& model, int T, int L, const vec<int>& C, const vec<vec<vec<int>>>& D) {
   vec<vec<vec<vec<vec<GRBVar>>>>> bw(T);
   for(int t = 0; t < T; ++t) {
      bw[t] = vec<vec<vec<vec<GRBVar>>>>(L);
      for(int k = 0; k < L; ++k) {
         bw[t][k] = vec<vec<vec<GRBVar>>>(C[k] + 1);
         for(int i = 0; i <= C[k]; ++i) {
            bw[t][k][i] = vec<vec<GRBVar>>(C[k + 1]);
            for(int j = 0; j < C[k + 1]; ++j) {
               bw[t][k][i][j] = vec<GRBVar>(D[k][i][j] + 1);
               for(int l = 0; l <= D[k][i][j]; ++l) {
                  bw[t][k][i][j][l] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("bw_{}_{}_{}_{}_{}", t, k, i, j, l));
               }
            }
         }
      }
   }
   return bw;
}

auto linear_activation_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   vec<vec<vec<GRBVar>>> z(T);
   for(int t = 0; t < T; ++t) {
      z[t] = vec<vec<GRBVar>>(L);
      for(int k = 0; k < L; ++k) {
         z[t][k] = vec<GRBVar>(C[k + 1]);
         for(int j = 0; j < C[k + 1]; ++j) {
            z[t][k][j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("z_{}_{}_{}", t, k, j));
         }
      }
   }
   return z;
}

auto total_error_variables(GRBModel& model, int T) {
   vec<GRBVar> E(T); 
   for(int t = 0; t < T; ++t) {
      E[t] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("E_{}", t));
   }
   return E;
}

auto output_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   vec<vec<GRBVar>> y(T);
   for(int t = 0; t < T; ++t) {
      y[t] = vec<GRBVar>(C[L]);
      for(int j = 0; j < C[L]; ++j) {
         y[t][j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("y_{}_{}", t, j));
      }
   }
   return y;
}

auto abs_error_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   vec<vec<GRBVar>> dy(T);
   for(int t = 0; t < T; ++t) {
      dy[t] = vec<GRBVar>(C[L]);
      for(int j = 0; j < C[L]; ++j) {
         dy[t][j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("dy_{}_{}", t, j));
      }
   }
   return dy;
}

auto error_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   vec<vec<GRBVar>> e(T);
   for(int t = 0; t < T; ++t) {
      e[t] = vec<GRBVar>(C[L]);
      for(int j = 0; j < C[L]; ++j) {
         e[t][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("e_{}_{}", t, j));
      }
   }
   return e;
}

auto input_variables(GRBModel& model, int T, const vec<int>& C, const vec<vec<double>>& px) {
   vec<vec<GRBVar>> x(T);
   for(int t = 0; t < T; ++t) {
      x[t] = vec<GRBVar>(C[0]);
      for(int i = 0; i < C[0]; ++i) {
         x[t][i] = model.addVar(px[t][i], px[t][i], 0, GRB_CONTINUOUS, format("x_{}_{}", t, i));
      }
   }
   return x;
}

auto activation_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   vec<vec<vec<GRBVar>>> a(T);
   for(int t = 0; t < T; ++t) {
      a[t] = vec<vec<GRBVar>>(L + 1);
      for(int k = 0; k <= L; ++k) {
         a[t][k] = vec<GRBVar>(C[k] + 1);
         for(int i = 0; i <= C[k]; ++i) {
            a[t][k][i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("a_{}_{}_{}", t, k, i));
         }
      }
   }
   return a;
}

auto bias_variables(GRBModel& model, int L, const vec<double>& used_bias) {
   vec<GRBVar> bias(L);
   for(int k = 0; k < L; ++k) {
      bias[k] = model.addVar(used_bias[k], used_bias[k], 0, GRB_CONTINUOUS, format("bias_{}", k));
   }
   return bias;
}

auto auxiliary_variables(GRBModel& model, int T, int L, const vec<int>& C) {
   vec<vec<vec<vec<GRBVar>>>> aux(T);
   for(int t = 0; t < T; ++t) {
      aux[t] = vec<vec<vec<GRBVar>>>(L);
      for(int k = 0; k < L; ++k) {
         aux[t][k] = vec<vec<GRBVar>>(C[k + 1]);
         for(int j = 0; j < C[k + 1]; ++j) {
            aux[t][k][j] = vec<GRBVar>(2);
            aux[t][k][j][0] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("aux_{}_{}_{}_A", t, k, j));
            aux[t][k][j][1] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, format("aux_{}_{}_{}_B", t, k, j));
         }
      }
   }
   return aux;
}

void absolute_error_constraints(GRBModel& model, const vec<vec<GRBVar>>& e, const vec<vec<GRBVar>>& y, const vec<vec<GRBVar>>& dy, int T, int L, const vec<int>& C, const vec<vec<double>>& ty) {
   for(int t = 0; t < T; ++t) {
      for(int j = 0; j < C[L]; ++j) {
         model.addConstr(y[t][j] - dy[t][j], GRB_EQUAL, ty[t][j], format("Difference_in_prediction_Neuron_{}_Case_{}", j, t));
         model.addGenConstrAbs(e[t][j], dy[t][j], format("Absolute_Error_Restriction_Neuron_{}_Case_{}", j, t));
      }
   }
}

void sum_error_constraints(GRBModel& model, const GRBVar& J, const vec<GRBVar>& E, const vec<vec<GRBVar>>& e, int T, int L, const vec<int>& C) {
   GRBLinExpr case_sum_error_expr;
   for(int t = 0; t < T; ++t) {
      GRBLinExpr sum_error_expr;
      for(int j = 0; j < C[L]; ++j) {
         sum_error_expr += e[t][j];
      }
      model.addConstr(sum_error_expr, GRB_EQUAL, E[t], format("Absolute_Error_Case_{}", t));
      case_sum_error_expr += E[t];
   }
   model.addConstr(case_sum_error_expr, GRB_EQUAL, J, format("Absolute_Total_Error"));
}

void max_error_objective(GRBModel& model, const GRBVar& J, const vec<GRBVar>& E, const vec<vec<GRBVar>>& e, int T, int L, const vec<int>& C) {
   GRBLinExpr case_sum_error_expr;
   for(int t = 0; t < T; ++t) {
      model.addGenConstrMax(E[t], e[t].data(), e[t].size(), 0, format("Max_Absolute_Error_Case_{}", t));
      case_sum_error_expr += E[t];
   }
   model.addConstr(case_sum_error_expr, GRB_EQUAL, J,format("Max_Absolute_Total_Error"));
}

void max_total_error_objective(GRBModel& model, const GRBVar& J, const vec<GRBVar>& E, const vec<vec<GRBVar>>& e, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      model.addGenConstrMax(E[t], e[t].data(), e[t].size(), 0, format("Max_Absolute_Error_Case_{}", t));
   }
   model.addGenConstrMax(J, E.data(), E.size(), 0, format("Max_Total_Absolute_Error"));
}

void output_layer_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& a, const vec<vec<GRBVar>>& y, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int j = 0; j < C[L]; ++j) {
         model.addConstr(y[t][j], GRB_EQUAL, a[t][L][j], format("Output_Layer_Neu{}_Case_{}", j, t));
      }
   }
}

void hardtanh_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int k, const vec<int>& C, double min_value = 0, double max_value = 1) {
   for(int t = 0; t < T; ++t) {
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMin(aux[t][k][j][0], &z[t][k][j], 1, max_value, format("L{}_Neu{}_Hardtanh_Case_{}", k, j, t));
         model.addGenConstrMax(a[t][k + 1][j], &aux[t][k][j][0], 1, min_value, format("L{}_Neu{}_Hardtanh_Case_{}", k, j, t));
      }
   }  
}

void hardsigmoid_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int k, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addConstr(z[t][k][j] / 6 + 0.5, GRB_EQUAL, aux[t][k][j][0], format("L{}_Neu{}_Hardsigmoid_Case_{}", k, j, t));
         model.addGenConstrMin(aux[t][k][j][1],&aux[t][k][j][0],1,1, format("L{}_Neu{}_Hardsigmoid_Case_{}", k, j, t));
         model.addGenConstrMax(a[t][k + 1][j],&aux[t][k][j][1],1,0, format("L{}_Neu{}_Hardsigmoid_Case_{}", k, j, t));
      }
   }  
}

void ReLU6_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int k, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, format("RELU6_Layer_{}_Neuron_{}_Case_{}", k, j, t));
         model.addGenConstrMin(a[t][k + 1][j],&aux[t][k][j][0],1,6, format("RELU6_Layer_{}_Neuron_{}_Case_{}", k, j, t));
      }
   }  
}

void ReLU_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int k, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMax(a[t][k + 1][j],&z[t][k][j],1,0, format("L{}_Neu{}_ReLU_Case_{}", k + 1, j, t));
      }
   }  
}

void Leaky_ReLU_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int k, const vec<int>& C, const vec<double>& negative_slope) {
   for(int t = 0; t < T; ++t) {
      for(int j = 0; j < C[k + 1]; ++j) {
         model.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, std::format("L{}_Neu{}_Leaky_ReLU_Case_{}", k, j, t));
         model.addGenConstrMin(aux[t][k][j][1],&z[t][k][j],1,0, std::format("L{}_Neu{}_Leaky_ReLU_Case_{}", k, j, t));
         model.addConstr(a[t][k + 1][j] == aux[t][k][j][0] + negative_slope[k] * aux[t][k][j][1], std::format("L{}_Neu{}_Leaky_ReLU_Case_{}", k, j, t));
      }
   }  
}

void bias_constraints(GRBModel& model, const vec<GRBVar>& layer_bias, const vec<vec<vec<GRBVar>>>& a, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         model.addConstr(layer_bias[k], GRB_EQUAL, a[t][k][C[k]], format("L{}_bias_Case_{}",k,t));
      }
   }
}

void binary_constraints(GRBModel& model, const vec<vec<vec<vec<GRBVar>>>>& b, const vec<vec<vec<vec<vec<GRBVar>>>>>& bw, const vec<vec<vec<GRBVar>>>& a, const vec<vec<vec<GRBVar>>>& z, int T, int L, const vec<int>& C, const vec<vec<vec<int>>>& D, const vec<vec<vec<int>>>& precis) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int i = 0; i <= C[k]; ++i) {
            for(int j = 0; j < C[k + 1]; ++j) {
               for(int l = 0; l <= D[k][i][j]; ++l) {
                  model.addGenConstrIndicator(b[k][i][j][l], 1, std::exp2(precis[k][i][j] - l) * a[t][k][i] - bw[t][k][i][j][l], GRB_EQUAL, 0, format("L{}_W{}{}_D{}_Descomp_Case_{}",k,i,j,l,t));
                  model.addGenConstrIndicator(b[k][i][j][l], 0, bw[t][k][i][j][l], GRB_EQUAL, 0, format("L{}_W{}{}_D{}_Descomp_Case_{}",k,i,j,l,t));
               }
            }
         }
      }
   }
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int j = 0; j < C[k + 1]; ++j) {
            GRBLinExpr neuron_linear_expr;
            for(int i = 0; i <= C[k]; ++i) {
               for(int l = 0; l < D[k][i][j]; ++l) {
                  neuron_linear_expr += bw[t][k][i][j][l];
               }
               neuron_linear_expr -= bw[t][k][i][j][D[k][i][j]];
            }
            model.addConstr(neuron_linear_expr, GRB_EQUAL, z[t][k][j], format("Lin_Output_L{}_Neu{}_Case_{}",k,j,t));
         }
      }
   }
}

void input_layer_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& a, const vec<vec<GRBVar>>& x, int T, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int i = 0; i < C[0]; ++i) {
         model.addConstr(a[t][0][i], GRB_EQUAL, x[t][i], format("Input_L_Case_{}_Neu{}", t, i));
      }
   }
}

GRBModel get_model(const GRBEnv& environment, int T, int L, const vec<int>& C, const vec<string>& AF, const vec<vec<vec<int>>>& D, const vec<vec<double>>& tx, const vec<vec<double>>& ty, const vec<vec<vec<int>>>& precis, const vec<double>& bias_w, const vec<double>& negative_slope, int optim_opt = 0) {
   GRBModel model(environment);
   GRBVar J = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, format("J"));
   auto E = total_error_variables(model,T);
   auto e = error_variables(model,T,L,C);
   auto dy = abs_error_variables(model,T,L,C);
   auto y = output_variables(model,T,L,C);
   auto aux = auxiliary_variables(model,T,L,C);
   auto z = linear_activation_variables(model,T,L,C);
   auto bw = pounded_activation_variables(model,T,L,C,D);
   auto b = layer_binary_variables(model,L,C,D);
   auto bias = bias_variables(model,L,bias_w);
   auto a = activation_variables(model,T,L,C);
   auto x = input_variables(model,T,C,tx);
   model.setObjective(GRBLinExpr(J), GRB_MINIMIZE); 

   switch(optim_opt) {
      case 1 :
         max_total_error_objective(model,J,E,e,T,L,C);
         break;
      case 2 :
         max_error_objective(model,J,E,e,T,L,C);
         break;
      default :
         sum_error_constraints(model,J,E,e,T,L,C);
   }
   absolute_error_constraints(model,e,y,dy,T,L,C,ty);
   output_layer_constraints(model,a,y,T,L,C);
   for(int k = 0; k < L; ++k) {
      if(AF[k].compare("ReLU") == 0) {
         ReLU_constraints(model,z,aux,a,T,k,C);
      }
      else if(AF[k].compare("ReLU6") == 0) {
         ReLU6_constraints(model,z,aux,a,T,k,C);
      }
      else if(AF[k].compare("Leaky_ReLU") == 0) {
         Leaky_ReLU_constraints(model,z,aux,a,T,k,C,negative_slope);
      }
      else if(AF[k].compare("Hardtanh") == 0) {
         hardtanh_constraints(model,z,aux,a,T,k,C,-1,1);
      }
      else if(AF[k].compare("Hardsigmoid") == 0) {
         hardsigmoid_constraints(model,z,aux,a,T,k,C);
      }
      else {
         ReLU_constraints(model,z,aux,a,T,k,C);
      }
   }
   bias_constraints(model,bias,a,T,L,C);
   binary_constraints(model,b,bw,a,z,T,L,C,D,precis);
   input_layer_constraints(model,a,x,T,C);
   return model;
}

auto read_fmatrix(const path& matrix_path) {
   fstream matrix_stream(matrix_path);
   string line, word;
   vec<vec<double>> matrix;
   std::getline(matrix_stream, line);
   while(std::getline(matrix_stream, line)) {
      stringstream line_stream(line);
      vec<double> row;
      std::getline(line_stream, word, ',');
      while(std::getline(line_stream, word, ',')) {
         row.push_back(std::stod(word));
      }
      matrix.push_back(row);
   }
   return matrix;
}

std::pair<vec<int>,vec<string>> process_arch(const vec<std::pair<int,string>>& arch, int C_0, int C_L) {
   int L = arch.size();
   vec<int> C(L + 1);
   vec<string> AF(L);
   std::transform(arch.begin(),arch.end() - 1,C.begin() + 1,[](const std::pair<int,string>& p) { return p.first; });
   C[0] = C_0;
   C[L] = C_L;
   std::transform(arch.begin(),arch.end(),AF.begin(),[](const std::pair<int,string>& p) { return p.second; });
   return {C, AF};
}

auto read_slope(const path& arch_path) {
   fstream arch_stream(arch_path);
   string line, word;
   vec<double> negative_slope;
   std::getline(arch_stream, line);
   while(std::getline(arch_stream, line)) {
      stringstream line_stream(line);
      try {
         std::getline(line_stream, word, ',');
         std::getline(line_stream, word, ',');
         std::getline(line_stream, word, ',');
         if(std::getline(line_stream, word, ',')) {
            negative_slope.push_back(std::stod(word));
         }
         else {
            negative_slope.push_back(0);
         }
      }
      catch(const std::invalid_argument& e) {
         negative_slope.push_back(0);
      }
   }
   return negative_slope;
}

auto read_arch(const path& arch_path) {
   fstream arch_stream(arch_path);
   string line, word;
   vec<std::pair<int,string>> arch;
   std::getline(arch_stream, line);
   while(std::getline(arch_stream, line)) {
      stringstream line_stream(line);
      std::pair<int,string> layer;
      std::getline(line_stream, word, ',');
      try {
         std::getline(line_stream, word, ',');
         layer.first = std::stoi(word);
      }
      catch(const std::invalid_argument& e) {
         layer.first = -1;
      }
      std::getline(line_stream, word, ',');
      layer.second = word;
      arch.push_back(layer);
   }
   return arch;
}

auto read_imatrix_list(const path& list_path) {
   fstream list_stream(list_path);
   string line, word;
   vec<vec<vec<int>>> list;
   std::getline(list_stream, line);
   while(std::getline(list_stream, line)) {
      stringstream line_stream(line);
      std::getline(line_stream, word, ',');
      std::getline(line_stream, word, ',');
      int n = std::stoi(word);
      std::getline(line_stream, word, ',');
      int m = std::stoi(word);
      vec<vec<int>> matrix(n,vec<int>(m));
      for(int i = 0; i < n; ++i) {
         for(int j = 0; j < m; ++j) {
            std::getline(line_stream, word, ',');
            matrix[i][j] = std::stoi(word);
         }
      }
      list.push_back(matrix);
   }
   return list;
}

auto read_fvector(const path& vector_path) {
   fstream vector_stream(vector_path);
   string line, word;
   vec<double> vector;
   std::getline(vector_stream, line);
   while(std::getline(vector_stream, line)) {
      stringstream line_stream(line);
      std::getline(line_stream, word, ',');
      std::getline(line_stream, word, ',');
      vector.push_back(std::stod(word));
   }
   return vector;
}

int main(int argc, const char* argv[]) try {
   // lectura de la entrada
   if(argc < 7) 
      return 0;
   path arch_path(argv[1]);
   path cases_path(argv[2]);
   path labels_path(argv[3]);
   path precision_path(argv[4]);
   path digits_path(argv[5]);
   path bias_path(argv[6]);
   string opt;
   if(argc >= 8) {
      opt = argv[7];
   }
   else {
      opt = "LP";
   }

   auto arch = read_arch(arch_path);
   auto cases = read_fmatrix(cases_path);
   auto labels = read_fmatrix(labels_path);
   auto precision = read_imatrix_list(precision_path);
   auto digits = read_imatrix_list(digits_path);
   auto bias = read_fvector(bias_path);
   auto negative_slope = read_slope(arch_path);
   int T = cases.size(), L = arch.size();
   int C_0 = cases[0].size(), C_L = labels[0].size();
   auto [C, AF] = process_arch(arch,C_0,C_L);

   // construcción del modelo
   
   GRBEnv ambiente;
   GRBModel modelo = get_model(ambiente,T,L,C,AF,digits,cases,labels,precision,bias,negative_slope);
   
   if(opt == "LP") {
      modelo.write(path(format("model.lp")));       // se puede quitar en caso de no desear generar el modelo en formato LP
   }
   else if(opt == "SOL") {
      // ------ resolución del modelo
      modelo.optimize( );
      if (modelo.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
         std::cout << "Solución encontrada\n";
         // ... examinar la solución en caso de requerirlo
         modelo.write(path(format("model.sol")));   // se puede quitar en caso de no desear generar el archivo .sol
      } else if (modelo.get(GRB_IntAttr_Status) == GRB_UNBOUNDED) {
         std::cout << "Modelo no acotado\n";
      } else if (modelo.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
         std::cout << "Modelo infactible\n";
         modelo.computeIIS( );
         modelo.write(path(format("model.ilp")));
      } else {
         std::cout << "Estado no manejado\n";
      }
   }
} catch (const GRBException& ex) {
   std::cout << ex.getMessage( ) << "\n";
}

// g++ programa.cpp -std=c++23 -lgurobi_c++ -lgurobi110 -o programa
// ./programa arch_path database_path labels_path
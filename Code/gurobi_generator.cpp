#include <format>
#include <iostream>
#include <gurobi_c++.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>

namespace fsys = std::filesystem;
using std::fstream;
using fsys::path;
using std::format;

template<typename T>
using vec = std::vector<T>;

auto layer_binary_variables(GRBModel& model, int L, const vec<int>& C, int D) {
   vec<vec<vec<vec<GRBVar>>>> b(L);
   for(int k = 0; k < L; ++k) {
      b[k] = vec<vec<vec<GRBVar>>>(C[k] + 1);
      for(int i = 0; i <= C[k]; ++i) {
         b[k][i] = vec<vec<GRBVar>>(C[k + 1]);
         for(int j = 0; j < C[k + 1]; ++j) {
            b[k][i][j] = vec<GRBVar>(D + 1);
            for(int l = 0; l <= D; ++l) {
               b[k][i][j][l] = model.addVar(0, 1, 0, GRB_BINARY, format("b_{}_{}_{}_{}",  k, i, j, l));
            }
         }
      }
   }
   return b;
}

auto pounded_activation_variables(GRBModel& model, int T, int L, const vec<int>& C, int D) {
   vec<vec<vec<vec<vec<GRBVar>>>>> bw(T);
   for(int t = 0; t < T; ++t) {
      bw[t] = vec<vec<vec<vec<GRBVar>>>>(L);
      for(int k = 0; k < L; ++k) {
         bw[t][k] = vec<vec<vec<GRBVar>>>(C[k] + 1);
         for(int i = 0; i <= C[k]; ++i) {
            bw[t][k][i] = vec<vec<GRBVar>>(C[k + 1]);
            for(int j = 0; j < C[k + 1]; ++j) {
               bw[t][k][i][j] = vec<GRBVar>(D + 1);
               for(int l = 0; l <= D; ++l) {
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

auto auxiliary_variables(GRBModel& model, int T, int L, const vec<int>& C, bool leaky = false) {
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

void hardtanh_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int L, const vec<int>& C, double min_value = 0, double max_value = 1) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int j = 0; j < C[k + 1]; ++j) {
            model.addGenConstrMin(aux[t][k][j][0], &z[t][k][j], 1, max_value, format("L{}_Neu{}_Hardtanh_Case_{}", k, j, t));
            model.addGenConstrMax(a[t][k + 1][j], &aux[t][k][j][0], 1, min_value, format("L{}_Neu{}_Hardtanh_Case_{}", k, j, t));
         }
      }
   }  
}

void hardsigmoid(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int j = 0; j < C[k + 1]; ++j) {
            model.addConstr(z[t][k][j] / 6 + 0.5, GRB_EQUAL, aux[t][k][j][0], format("L{}_Neu{}_Hardsigmoid_Case_{}", k, j, t));
            model.addGenConstrMin(aux[t][k][j][1],&aux[t][k][j][0],1,1, format("L{}_Neu{}_Hardsigmoid_Case_{}", k, j, t));
            model.addGenConstrMax(a[t][k + 1][j],&aux[t][k][j][1],1,0, format("L{}_Neu{}_Hardsigmoid_Case_{}", k, j, t));
         }
      }
   }  
}

void ReLU6_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int j = 0; j < C[k + 1]; ++j) {
            model.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, format("RELU6_Layer_{}_Neuron_{}_Case_{}", k, j, t));
            model.addGenConstrMin(a[t][k + 1][j],&aux[t][k][j][0],1,6, format("RELU6_Layer_{}_Neuron_{}_Case_{}", k, j, t));
         }
      }
   }  
}

void ReLU_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int L, const vec<int>& C) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int j = 0; j < C[k + 1]; ++j) {
            model.addGenConstrMax(a[t][k + 1][j],&z[t][k][j],1,0, format("L{}_Neu{}_ReLU_Case_{}", k + 1, j, t));
         }
      }
   }  
}

void Leaky_ReLU_constraints(GRBModel& model, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, int T, int L, const vec<int>& C, const vec<vec<double>>& negative_slope) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int j = 0; j < C[k + 1]; ++j) {
            model.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, std::format("L{}_Neu{}_Leaky_ReLU_Case_{}", k, j, t));
            model.addGenConstrMin(aux[t][k][j][1],&z[t][k][j],1,0, std::format("L{}_Neu{}_Leaky_ReLU_Case_{}", k, j, t));
            model.addConstr(a[t][k + 1][j] == aux[t][k][j][0] + negative_slope[k][j] * aux[t][k][j][1], std::format("L{}_Neu{}_Leaky_ReLU_Case_{}", k, j, t));
         }
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

void binary_constraints(GRBModel& model, const vec<vec<vec<vec<GRBVar>>>>& b, const vec<vec<vec<vec<vec<GRBVar>>>>>& bw, const vec<vec<vec<GRBVar>>>& a, const vec<vec<vec<GRBVar>>>& z, int T, int L, const vec<int>& C, int D, const vec<vec<vec<int>>>& precis) {
   for(int t = 0; t < T; ++t) {
      for(int k = 0; k < L; ++k) {
         for(int i = 0; i <= C[k]; ++i) {
            for(int j = 0; j < C[k + 1]; ++j) {
               for(int l = 0; l <= D; ++l) {
                  model.addGenConstrIndicator(b[k][i][j][l], 1, std::exp2(l - precis[k][i][j]) * a[t][k][i] - bw[t][k][i][j][l], GRB_EQUAL, 0, format("L{}_W{}{}_D{}_Descomp_Case_{}",k,i,j,l,t));
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
               for(int l = 0; l < D; ++l) {
                  neuron_linear_expr += bw[t][k][i][j][l];
               }
               neuron_linear_expr -= bw[t][k][i][j][D];
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

GRBModel get_model(const GRBEnv& environment, int T, int L, const vec<int>& C, int D, const vec<vec<double>>& tx, const vec<vec<double>>& ty, const vec<vec<vec<int>>>& precis, const vec<double>& bias_w) {
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

   sum_error_constraints(model,J,E,e,T,L,C);
   //max_error_objective(model,J,E,e,T,L,C);
   //max_total_error_objective(model,J,E,e,T,L,C);
   absolute_error_constraints(model,e,y,dy,T,L,C,ty);
   output_layer_constraints(model,a,y,T,L,C);
   ReLU_constraints(model,z,aux,a,T,L,C);
   //ReLU6_constraints(model,z,aux,a,T,L,C);
   //Leaky_ReLU_constraints(model,z,aux,a,T,L,C,negative_slope);
   //hardtanh_constraints(model,z,aux,a,T,L,C,-1,1);
   //hardsigmoid_constraints(model,z,aux,a,T,L,C);
   bias_constraints(model,bias,a,T,L,C);
   binary_constraints(model,b,bw,a,z,T,L,C,D,precis);
   input_layer_constraints(model,a,x,T,C);
   return model;
}

int main(int argc, const char* argv[]) try {
   // lectura de la entrada
   path arch_path(argv[1]);
   path database_path(argv[2]);
   path labels_path(argv[3]);

   fstream arch_stream(arch_path);
   int T;
   arch_stream >> T;
   int L;
   arch_stream >> L;
   vec<int> C(L + 1);
   for(auto& layer_input : C) {
      arch_stream >> layer_input;
   }

   vec<vec<vec<int>>> precis(L);
   for(int k = 0; k < L; ++k) {
      precis[k] = vec<vec<int>>(C[k] + 1);
      for(auto& expon_layer : precis[k]) {
         expon_layer = vec<int>(C[k + 1]);
         for(auto& expon : expon_layer) {
            expon = 1;
         }
      }
   }

   fstream database_stream(database_path);
   vec<vec<double>> tx(T);
   for(auto& tcase : tx) {
      tcase = vec<double>(C[0]);
      for(auto& attribute : tcase) {
         database_stream >> attribute;
      }
   }

   fstream labels_stream(labels_path);
   vec<vec<double>> ty(T);
   for(auto& tcase : ty) {
      tcase = vec<double>(C[L]);
      for(auto& attribute : tcase) {
         labels_stream >> attribute;
      }
   }
   vec<double> bias_w(L,1);
   int D = 16;
   
   // construcción del modelo
   
   GRBEnv ambiente;
   GRBModel modelo = get_model(ambiente,T,L,C,D,tx,ty,precis,bias_w);
   
   // ------ resolución del modelo
   modelo.optimize( );
   modelo.write(path(format("model.lp")));       // se puede quitar en caso de no desear generar el modelo en formato LP
   
   /**/
   if (modelo.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
      std::cout << "Solución encontrada\n";
      // ... examinar la solución en caso de requerirlo
      modelo.write(format("{}.sol", argv[0]));   // se puede quitar en caso de no desear generar el archivo .sol
   } else if (modelo.get(GRB_IntAttr_Status) == GRB_UNBOUNDED) {
      std::cout << "Modelo no acotado\n";
   } else if (modelo.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
      std::cout << "Modelo infactible\n";
      modelo.computeIIS( );
      modelo.write(format("{}.ilp", argv[0]));
   } else {
      std::cout << "Estado no manejado\n";
   }
   /**/
} catch (const GRBException& ex) {
   std::cout << ex.getMessage( ) << "\n";
}

// g++ programa.cpp -std=c++23 -lgurobi_c++ -lgurobi110 -o programa
// ./programa arch_path database_path labels_path
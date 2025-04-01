#include <format>
#include <iostream>
#include <gurobi_c++.h>
#include <vector>
#include <cmath>

using vec = std::vector;

auto layer_binary_variables(int L, const vec<int>& C, int D) {
    vec<vec<vec<vec<GRBVar>>>> b = vec<vec<vec<vec<GRBVar>>>>(L);
    for(int k = 0; k < L; ++k) {
        b[k] = vec<vec<vec<GRBVar>>>(C[k]);
        for(int i = 0; i <= C[k]; ++i) {
            b[k][i] = vec<vec<GRBVar>>(C[k + 1]);
            for(int j = 0; j < C[k + 1]; ++j) {
                b[k][i][j] = vec<GRBVar>(D + 1);
                for(int l = 0; l <= D; ++l) {
                    b[k][i][j][l]  = modelo.addVar(0, 1, 0, GRB_BINARY, std::format("b_{}_{}_{}_{}",  k, i, j, l));
                }
            }
        }
    }
    return b;
}

auto pounded_activation_variables(int T, int L, const vec<int>& C, int D) {
    vec<vec<vec<vec<vec<GRBVar>>>>> bw = vec<vec<vec<vec<vec<GRBVar>>>>>(T);
    for(int t = 0; t < T; ++t) {
        bw[t] = vec<vec<vec<vec<GRBVar>>>>(L);
        for(int k = 0; k < L; ++k) {
            bw[t][k] = vec<vec<vec<GRBVar>>>(C[k]);
            for(int i = 0; i <= C[k]; ++i) {
                bw[t][k][i] = vec<vec<GRBVar>>(C[k + 1]);
                for(int j = 0; j < C[k + 1]; ++j) {
                    bw[t][k][i][j] = vec<GRBVar>(D + 1);
                    for(int l = 0; l <= D; ++l) {
                        bw[t][k][i][j][l] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("bw_{}_{}_{}_{}_{}", t, k, i, j, l));
                    }
                }
            }
        }
    }
    return bw;
}

auto linear_activation_variables(int T, int L, const vec<int>& C) {
    vec<vec<vec<GRBVar>>> z = vec<vec<vec<GRBVar>>>(T);
    for(int t = 0; t < T; ++t) {
        z[t] = vec<vec<vec<vec<GRBVar>>>>(L);
        for(int k = 0; k < L; ++k) {
            z[t][k] = vec<vec<vec<GRBVar>>>(C[k + 1]);
            for(int j = 0; j < C[k + 1]; ++j) {
                z[t][k][j] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("z_{}_{}_{}", t, k, j));
            }
        }
    }
    return z;
}

auto total_error_variables(int T) {
    vec<GRBVar> E = vec<GRBVar>(T); 
    for(int t = 0; t < T; ++t) {
        E[t] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("E_{}", t));
    }
    return E;
}

auto output_variables(int T, int L, const vec<int>& C) {
    vec<vec<GRBVar>> y = vec<vec<GRBVar>>(T);
    for(int t = 0; t < T; ++t) {
        y[t] = vec<GRBVar>(C[L]);
        for(int j = 0; j < C[L]; ++j) {
            y[t][j] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("y{}_{}", t, j));
        }
    }
    return y;
}

auto abs_error_variables(int T, int L, const vec<int>& C) {
    vec<vec<vec<GRBVar>>> y_abs = vec<vec<vec<GRBVar>>>(T);
    for(int t = 0; t < T; ++t) {
        y_abs[t] = vec<vec<GRBVar>>(C[L]);
        for(int j = 0; j < C[L]; ++j) {
            y_abs[t][j] = vec<GRBVar>(2);
            y_abs[t][j][0] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("y_A_{}_{}", t, j));
            y_abs[t][j][1] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("y_B_{}_{}", t, j));
        }
    }
    return y_abs;
}

auto error_variables(int T, int L, const vec<int>& C) {
    vec<vec<GRBVar>> e = vec<vec<GRBVar>>(T);
    for(int t = 0; t < T; ++t) {
        e[t] = vec<GRBVar>(C[L]);
        for(int j = 0; j < C[L]; ++j) {
            e[t][j] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("e_{}_{}", t, j));
        }
    }
    return e;
}

auto input_variables(int T, const vec<int>& C, const vec<vec<double>>& px) {
    vec<vec<GRBVar>> x = vec<vec<GRBVar>>(T);
    for(int t = 0; t < T; ++t) {
        x[t] = vec<GRBVar>(C[0]);
        for(int i = 0; i < C[0]; ++i) {
            x[t][i] = modelo.addVar(px[t][i], px[t][i], 0, GRB_CONTINUOUS, std::format("x{}_{}", t, i));
        }
    }
    return x;
}

auto activation_variables(int T, int L, const vec<int>& C) {
    vec<vec<vec<GRBVar>>> a = vec<vec<vec<GRBVar>>>(T);
    for(int t = 0; t < T; ++t) {
        a[t] = vec<vec<GRBVar>>(L + 1);
        for(int k = 0; k <= L; ++k) {
            a[t][k] = vec<vec<GRBVar>>(C[k]);
            for(int i = 0; i <= C[k]; ++i) {
                a[t][k][i] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("a_{}_{}_{}", t, k, i));
            }
        }
    }
    return a;
}

auto bias_variables(int L, const vec<int>& C, const vec<vec<double>>& used_bias) {
    vec<vec<GRBVar>> bias = vec<vec<GRBVar>>(L + 1);
    for(int k = 0; k < L; ++k) {
        bias[k] = vec<GRBVar>(C[k]);
        for(int i = 0; i < C[k]; ++i) {
            bias[k][i] = modelo.addVar(used_bias[k][i], used_bias[k][i], 0, GRB_CONTINUOUS, std::format("bias_{}_{}", k, i));
        }
    }
    return bias;
}

auto auxiliary_variables(int T, int L, const vec<int>& C, bool leaky = false) {
    vec<vec<vec<vec<GRBVar>>>> aux(C[k]);
    for(int t = 0; t < T; ++t) {
        aux[t] = vec<vec<vec<GRBVar>>>(L);
        for(int k = 0; k < L; ++k) {
            aux[t][k] = vec<vec<GRBVar>>(C[k + 1]);
            for(int j = 0; j < C[k + 1]; ++j) {
                aux[t][k][j] = vec<GRBVar>(2);
                aux[t][k][j][0] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("aux_{}_{}_{}_A", t, k, j));
                aux[t][k][j][1] = modelo.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("aux_{}_{}_{}_B", t, k, j));
            }
        }
    }
    return aux;
}

void absolute_error_constraints(GRBModel& modelo, const vec<vec<GRBVar>>& e, const vec<vec<GRBVar>>& y, const vec<vec<vec<GRBVar>>>& y_abs, const vec<vec<GRBVar>>& e, int T, int L, const vec<int>& C, const vec<vec<double>>& ty) {
    for(int t = 0; t < T; ++t) {
        for(int j = 0; j < C[L]; ++j) {
            modelo.addConstr(y_abs[t][j][0] == y[t][j] - ty[t][j], std::format("Difference_in_prediction_Neuron_{}_Case_{}", j, t));
            modelo.addConstr(y_abs[t][j][1] == ty[t][j] - y[t][j], std::format("Difference_in_prediction_Neuron_{}_Case_{}", j, t));
            modelo.addGenConstrMax(e[t][j], &y_abs[t][0], 2, 0, std::format("Absolute_Error_Restriction_Neuron_{}_Case_{}", j, t));
        }
    }
}

void sum_error_objetive(GRBModel& modelo, const GRBVar& J, const vec<GRBVar>& E, const vec<vec<GRBVar>>& e, int T, int L, const vec<int>& C) {
    GRBLinExpr case_sum_error_expr;
    for(int t = 0; t < T; ++t) {
        GRBLinExpr sum_error_expr;
        for(int j = 1; j <= C[L]; ++j) {
            sum_error_expr += e[t][j];
        }
        modelo.addConstr(E[t] == sum_error_expr, std::format("Absolute_Error_For_Case_{}", t));
        case_sum_error_expr += E[t];
    }
    modelo.addConstr(J == objective_expr, std::format("Absolute_Total_Error"));
    modelo.setObjective(J, GRB_MINIMIZE); 
}

void max_error_objetive(GRBModel& modelo, const vec<GRBVar>& E, const vec<vec<GRBVar>>& e, int L, const vec<int>& C, int T) {
    GRBLinExpr case_sum_error_expr;
    for(int t = 0; t < T; ++t) {
        GRBLinExpr sum_error_expr;
        for(int j = 1; j <= C[L]; ++j) {
            sum_error_expr += e[t][j];
        }
        modelo.addGenConstrMax(E[t],e,e.size,0,std::format("Max_Absolute_Error_Case_{}", t));
        case_sum_error_expr += E[t];
    }
    modelo.addConstr(J == objective_expr, std::format("Max_Absolute_Total_Error"));
    modelo.setObjective(J, GRB_MINIMIZE); 
}

void max_total_error_objetive(GRBModel& modelo, const vec<GRBVar>& E, const vec<vec<GRBVar>>& e, int L, const vec<int>& C, int T) {
    GRBLinExpr case_sum_error_expr;
    for(int t = 0; t < T; ++t) {
        GRBLinExpr sum_error_expr;
        for(int j = 1; j <= C[L]; ++j) {
            sum_error_expr += e[t][j];
        }
        modelo.addGenConstrMax(E[t],e,e.size,0,std::format("Max_Absolute_Error_Case_{}", t));
        case_sum_error_expr += E[t];
    }
    modelo.addGenConstrMax(J,E.data(),E.size(),0,std::format("Max_Total_Absolute_Error", t));
    modelo.setObjective(J, GRB_MINIMIZE); 
}

void output_layer_constraints(GRBModel& modelo, const vec<vec<vec<GRBVar>>>& a, const vec<vec<GRBVar>>& y, int T, int L, const vec<int>& C) {
    for(int t = 0; t < T; ++t) {
        for(int j = 0; j < C[L]; ++j) {
            modelo.addConstr(y[t][j] == a[t][L][j],std::format("Output_Layer_Neuron_{}_Case_{}", j, t));
        }
    }
}

void hardtanh_constraints(GRBModel& modelo, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, double min_value = 0, double max_value = 1) {
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            for(int j = 0; j < C[k + 1]; ++j) {
                modelo.addGenConstrMin(aux[t][k][j][0],&z[t][k][j],1,max_value, std::format("Hardtanh_Layer_{}_Neuron_{}_Case_{}", k, j, t));
                modelo.addGenConstrMax(a[t][k][j],&aux[t][k][j][0],1,min_value, std::format("Hardtanh_Layer_{}_Neuron_{}_Case_{}", k, j, t));
            }
        }
    }  
}

void hardsigmoid(GRBModel& modelo, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a) {
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            for(int j = 0; j < C[k + 1]; ++j) {
                modelo.addConstr(aux[t][k][j][0] == z[t][k][j] / 6 + 0.5, std::format("Hardsigmoid_Layer_{}_Neuron_{}_Case_{}", k, j, t));
                modelo.addGenConstrMin(aux[t][k][j][1],&aux[t][k][j][0],1,1, std::format("Hardsigmoid_Layer_{}_Neuron_{}_Case_{}", k, j, t));
                modelo.addGenConstrMax(a[t][k][j],&aux[t][k][j][1],1,0, std::format("Hardsigmoid_Layer_{}_Neuron_{}_Case_{}", k, j, t));
            }
        }
    }  
}

void ReLU6_constraints(GRBModel& modelo, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a) {
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            for(int j = 0; j < C[k + 1]; ++j) {
                modelo.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, std::format("RELU6_Layer_{}_Neuron_{}_Case_{}", k, j, t));
                modelo.addGenConstrMin(a[t][k][j],&a[t][k][i][0],1,6, std::format("RELU6_Layer_{}_Neuron_{}_Case_{}", k, j, t));
            }
        }
    }  
}

void ReLU_constraints(GRBModel& modelo, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a) {
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            for(int j = 0; j < C[k + 1]; ++j) {
                modelo.addGenConstrMax(a[t][k][j],&z[t][k][j],1,0, std::format("RELU_Layer_{}_Neuron_{}_Case_{}", k, j, t));
            }
        }
    }  
}

void Leaky_ReLU_constraints(GRBModel& modelo, const vec<vec<vec<GRBVar>>>& z, const vec<vec<vec<vec<GRBVar>>>>& aux, const vec<vec<vec<GRBVar>>>& a, const vec<vec<double>>& negative_slope) {
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            for(int j = 0; j < C[k + 1]; ++j) {
                modelo.addGenConstrMax(aux[t][k][j][0],&z[t][k][j],1,0, std::format("Leaky_RELU_Layer_{}_Neuron_{}_Case_{}", k, j, t));
                modelo.addGenConstrMin(aux[t][k][i][1],&z[t][k][i],1,0, std::format("Leaky_RELU_Layer_{}_Neuron_{}_Case_{}", k, j, t));
                modelo.addConstr(a[t][k][j] == aux[t][k][j][0] + negative_slope[k][j] * aux[t][k][j][1], std::format("Leaky_RELU_Layer_{}_Neuron_{}_Case_{}", k, j, t));
            }
        }
    }  
}

void bias_constraints(GRBModel& modelo, const vec<GRBVar>& bias, const vec<vec<vec<GRBVar>>>& a, int T, int L, const vec<int>& C) {
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            modelo.addConstr(bias[k] == a[t][k][C[k]], std::format(""));
        }
    }
}

void binary_constraints(GRBModel& modelo, const vec<vec<vec<vec<GRBVar>>>>& b, const vec<vec<vec<vec<vec<GRBVar>>>>>& bw, const vec<vec<vec<GRBVar>>>& a, const vec<vec<vec<GRBVar>>>& z, int T, int L, const vec<int>& C, int D, const vec<vec<double>>& coef) {
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            for(int i = 0; i <= C[k]; ++i) {
                for(int j = 0; i < C[k + 1]; ++j) {
                    for(int l = 0; l <= D; ++l) {
                        modelo.addGenConstrIndicator(b[k][i][j][l], 1, std::exp2(l) * coef[k][i] * a[t][k][i] - bw[t][k][j][l], GRB_EQUAL, 0, std::format(""));
                    }
                }
            }
        }
    }
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < L; ++k) {
            for(int j = 0; i < C[k + 1]; ++j) {
                GRBLinExpr neuron_linear_expr;
                for(int l = 0; l < D; ++l) {
                    neuron_linear_expr += bw[t][k][j][l];
                }
                neuron_linear_expr -= bw[t][k][j][D];
                modelo.addConstr(z[t][k][j] == neuron_linear_expr, std::format(""));
            }
        }
    }
}

void input_layer_constraints(GRBModel& modelo, const vec<vec<vec<GRBVar>>>& a, const vec<vec<GRBVar>>& x, int T, const vec<int>& C) {
    for(int t = 0; t < T; ++t) {
        for(int i = 0; i < C[0]; ++i) {
            modelo.addConstr(a[t][0][i] == x[t][i],std::format("Input_Layer_Neuron_{}_Case_{}", i, t));
        }
    }
}

GRBModel get_model(const GRBEnv& environment, int T, int L, const vec<int>& C, int D, const vec<vec<double>>& tx, const vec<vec<double>>& ty, const vec<vec<double>> coef, const vec<double>& bias_w) {
    GRBModel model(environment);
    GRBVar J = model.addVar(NULL, NULL, 0, GRB_CONTINUOUS, std::format("J"));
    auto E = total_error_variables(T);
    auto e = error_variables(T,L,C);
    auto y_abs = abs_error_variables(T,L,C);
    auto y = output_variables(T,L,C);
    auto aux = auxiliary_variables(T,L,C);
    auto z = linear_activation_variables(T,L,C);
    auto bw = pounded_activation_variables(T,L,C,D);
    auto b = layer_binary_variables(L,C,D);
    auto bias = bias_variables(L,C,bias_w);
    auto a = activation_variables(T,L,C);
    auto x = input_variables(T,C,tx);
    
    sum_error_objective(model,J,E,e,T,L,C);
    //max_error_objective(model,J,E,e,T,L,C);
    //max_total_error_objective(model,J,E,e,T,L,C);
    absolute_error_constraints(model,e,y_abs,y,T,L,C,ty);
    output_layer_constraints(model,a,y,T,L,C);
    ReLU_constraints(model,z,aux,a,T,L,C);
    bias_constraints(model,bias,a,T,L,C);
    binary_constraints(model,b,bw,a,z,T,L,C,D,coef);
    input_layer_constraints(model,a,x,T,C);
    return model;
}

int main(int argc, const char* argv[]) try {)
   // lectura de la entrada
   std::cin >> L;
   vec<int> C(L + 1);
   for(auto& layer_input : C) {
    std::cin >> layer_input;
   }
   vec<vec<double>> coef(L);
   for(auto& coef_layer : coef) {
    coef_layer = vec<double>(C[k] + 1);
    for(auto& coefficient : coef_layer) {
        coefficient = 1;
    }
   }
   std::cin >> T;
   vec<vec<double>> tx(T);
   for(auto& tcase : tx) {
    tcase = vec<double>(C[0]);
    for(auto& attribute : tcase) {
        std::cin >> attribute;
    }
   }
   vec<vec<double>> ty(T);
   for(auto& tcase : ty) {
    tcase = vec<double>(C[L]);
    for(auto& attribute : tcase) {
        std::cin >> attribute;
    }
   }
   vec<double> bias_w(L,1);

   // construcción del modelo
   
   GRBEnv ambiente;;
   GRBModel modelo = get_model(ambiente,T,L,C,D,tx,ty,coef,bias_w);
   //get_model(ambiente,T,L,C,D,tx,ty,coef,bias_w,true,negative_slope);

   // ------ resolución del modelo
   modelo.optimize( );
   modelo.write(std::format("{}.lp", argv[0]));       // se puede quitar en caso de no desear generar el modelo en formato LP

   if (modelo.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
      std::cout << "Solución encontrada\n";
      // ... examinar la solución en caso de requerirlo
      modelo.write(std::format("{}.sol", argv[0]));   // se puede quitar en caso de no desear generar el archivo .sol
   } else if (modelo.get(GRB_IntAttr_Status) == GRB_UNBOUNDED) {
      std::cout << "Modelo no acotado\n";
   } else if (modelo.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
      std::cout << "Modelo infactible\n";
      modelo.computeIIS( );
      modelo.write(std::format("{}.ilp", argv[0]));
   } else {
      std::cout << "Estado no manejado\n";
   }
} catch (const GRBException& ex) {
   std::cout << ex.getMessage( ) << "\n";
}

// g++ programa.cpp -std=c++23 -lgurobi_c++ -lgurobi110 -o programa
// ./programa
import random
import math
import numpy
import matplotlib.pyplot as plt

def monte_carlo_sampling(N):
    iterations=(2**N)*10
    unique_configs=[]
    sampled_configs=[]
    for j in range(iterations):
        config = []
        for i in range(N):
            x=random.uniform(0,1)
            if x<=0.5:
                config.append(1)
            elif 0.5 < x <= 1:
                config.append(-1)
        if config not in sampled_configs:
            unique_configs.append(config)
        sampled_configs.append(config)
    return unique_configs
def monte_carlo_avg_m(unique_configs,N,T,J,h):
    z=0
    avg_m=0
    for unique_config in unique_configs:
        sum_xy=0
        for n in range(N-1):
            sum_xy+= unique_config[n]*unique_config[n+1]
        occurence=math.exp(-(-J*sum_xy- h*sum(unique_config))/T)
        z+=occurence
        avg_m+=occurence*sum(unique_config)
    return (avg_m/z)/N

def exact_m(N,T,J,h):
    lambda_1=(math.exp(J/T)) * (math.cosh(h/T) - math.sqrt( math.pow(math.sinh(h/T),2) + math.exp(-4*J/T)))
    lambda_2=(math.exp(J/T)) * (math.cosh(h/T) + math.sqrt( math.pow(math.sinh(h/T),2) + math.exp(-4*J/T)))
    der_1= math.sinh(h/T) - math.pow((math.pow(math.sinh(h/T),2) + math.exp(-4*J/T)),-1/2) * math.sinh(h/T) * math.cosh(h/T)
    der_2= math.sinh(h/T) + math.pow((math.pow(math.sinh(h/T),2) + math.exp(-4*J/T)),-1/2) * math.sinh(h/T) * math.cosh(h/T)
    avg_m= (math.exp(J/T) * (math.pow(lambda_1,N-1) * der_1 + math.pow(lambda_2,N-1) * der_2 )) / (math.pow(lambda_1,N)+ math.pow(lambda_2,N))
    return avg_m
def exact_m_infinite_N(T,J,h):
    lambda_2 = (math.exp(J / T)) * (math.cosh(h / T) + math.sqrt(math.pow(math.sinh(h / T), 2) + math.exp(-4 * J / T)))
    der_2 = math.sinh(h / T) + math.pow((math.pow(math.sinh(h / T), 2) + math.exp(-4 * J / T)), -1 / 2) * math.sinh(h / T) * math.cosh(h / T)
    avg_m=(math.exp(J/T) *der_2)/lambda_2
    return avg_m

def m_fixed_h(N_range,T,J,h):
    m_fixed_h_list_monte_carlo=[]
    m_fixed_h_list_exact=[]
    for N in N_range:
        unique_configs=monte_carlo_sampling(N)
        m_fixed_h_list_monte_carlo.append(monte_carlo_avg_m(unique_configs,N,T,J,h))
        m_fixed_h_list_exact.append(exact_m(N,T,J,h))
    return m_fixed_h_list_monte_carlo, m_fixed_h_list_exact

def m_fixed_N(h_range,N,T,J):
    unique_configs=monte_carlo_sampling(N)
    m_fixed_N_list_monte_carlo=[]
    m_fixed_N_list_exact=[]
    m_list_exact_infinite_N=[]
    for h in h_range:
        m_fixed_N_list_monte_carlo.append(monte_carlo_avg_m(unique_configs,N,T,J,h))
        m_fixed_N_list_exact.append(exact_m(N, T, J, h))
        m_list_exact_infinite_N.append(exact_m_infinite_N(T,J,h))
    return m_fixed_N_list_monte_carlo, m_fixed_N_list_exact,m_list_exact_infinite_N


def main():
    choice=eval(input("Press 1 for fixed h or 2 for fixed N : "))
    if choice == 1:
        J = 1
        T = 1
        h = 1
        N_range = list(range(2, 15 + 1))
        m_fixed_h_list_monte_carlo, m_fixed_h_list_exact  = m_fixed_h(N_range, T, J, h)
        plt.plot(N_range,m_fixed_h_list_monte_carlo,'ro')
        plt.plot(N_range,m_fixed_h_list_exact,'bo')
        plt.xlabel('N')
        plt.ylabel("<m>")
        plt.legend(['Monte Carlo Solution', "Exact Solution"])
        plt.title("Plot for Fixed h (h=1,T=1,J=1)")
        plt.show()
    if choice == 2:
        J = 1
        T = 1
        N=10
        h_range = numpy.arange(-1, 1 + 0.1, 0.1)
        m_fixed_N_list_monte_carlo,m_fixed_N_list_exact,m_list_exact_infinite_N=m_fixed_N(h_range,N,T,J)
        plt.plot(h_range,m_fixed_N_list_monte_carlo,'ro')
        plt.plot(h_range,m_fixed_N_list_exact,'bo')
        plt.plot(h_range, m_list_exact_infinite_N, 'go')
        plt.xlabel('h')
        plt.ylabel("<m>")
        plt.legend(['Monte Carlo Solution', "Exact Solution finite N","Exact solution infinite N"])
        plt.title("Plot for fixed N (J=1,N=10,T=1)")
        plt.show()

main()
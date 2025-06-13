
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.integrate import quad
from scipy.optimize import minimize
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols


class Heston:


    def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

        # constants
        a = kappa*theta
        b = kappa+lambd

        # common terms w.r.t phi
        rspi = rho*sigma*phi*1j

        # define d parameter given phi and b
        d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

        # define g parameter given phi, b and d
        g = (b-rspi+d)/(b-rspi-d)

        # calculate characteristic function by components
        exp1 = np.exp(r*phi*1j*tau)
        term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
        exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)
        
        return exp1*term2*exp2
    
    def integrand(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r, K):
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
        numerator = np.exp(r*tau)* Heston.heston_charfunc(phi-1j,*args) - K* Heston.heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)
        return numerator/denominator
    
    def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

        P, umax, N = 0, 100, 10000
        dphi=umax/N #dphi is width

        for i in range(1,N):
            # rectangular integration
            phi = dphi * (2*i + 1)/2 # midpoint to calculate height
            numerator = np.exp(r*tau)* Heston.heston_charfunc(phi-1j,*args) - K * Heston.heston_charfunc(phi,*args)
            denominator = 1j*phi*K**(1j*phi)

            P += dphi * numerator/denominator

        price = np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)

        return np.maximum(price, 0.0)
    
    def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

        real_integral, err = np.real( quad(Heston.integrand, 0, 100, args=args) )

        price = (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

        return np.maximum(price, 0.0)
    

    def SqErr(x, data, S0, use_penalty=False, x0_ref=None, alpha=0.0):

        v0, kappa, theta, sigma, rho, lambd = x

        # Prix modèle Heston
        prices_model = Heston.heston_price_rec(
            S0,
            data["strike"].to_numpy(),
            v0, kappa, theta, sigma, rho, lambd,
            data["maturity"].to_numpy(),
            data["rate"].to_numpy()
        )

        errors = ((data["price"].to_numpy() - prices_model) ** 2).mean()

        # Terme de pénalité si activé
        penalty = 0
        if use_penalty and x0_ref is not None:
            penalty = alpha * np.sum((np.array(x) - np.array(x0_ref)) ** 2)

        return errors + penalty
    


    def calibration(df, S0 = 218.27):



        yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
        yeilds = np.array([4.38, 4.33, 4.34, 4.29, 4.05, 3.99, 4.01, 4.09, 4.21, 4.32, 4.66, 4.62]).astype(float)/100

        #NSS model calibrate
        curve_fit, status = calibrate_nss_ols(yield_maturities,yeilds)





        market_prices = {}


        grouped = df.groupby("date_expiration")

        for expiration, group in grouped:
            expiration_str = expiration.strftime("%Y-%m-%d") 
            
            market_prices[expiration_str] = {}


            market_prices[expiration_str]['strike'] = group["strike"].tolist()
            market_prices[expiration_str]['price'] = group["price"].tolist()



        prices = []
        maturities = []
        strikes = []

        for date_str, v in market_prices.items():
            # Calcul de la maturité en années
            maturity = (dt.strptime(date_str, '%Y-%m-%d') - dt.strptime("2025-03-23", "%Y-%m-%d")).days / 365.25
            
            # On parcourt tous les strikes disponibles (pas d’intersection)
            for i, strike_i in enumerate(v['strike']):
                prices.append(v['price'][i])
                strikes.append(strike_i)
                maturities.append(maturity)


        df_prices = pd.DataFrame({
            'maturity': maturities,
            'strike': strikes,
            'price': prices
        })


        volSurface = df_prices.pivot_table(
            index='maturity',   
            columns='strike',   
            values='price'      
        )




        volSurfaceLong = volSurface.melt(ignore_index=False).reset_index()
        volSurfaceLong.columns = ['maturity', 'strike', 'price']


        volSurfaceLong.dropna(subset=["price"], inplace=True)


        volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(curve_fit)

  
        r = volSurfaceLong['rate'].to_numpy('float')
        K = volSurfaceLong['strike'].to_numpy('float')
        tau = volSurfaceLong['maturity'].to_numpy('float')
        P = volSurfaceLong['price'].to_numpy('float')


        params = {"v0": {"x0": 0.1, "lbub": [1e-3,0.1]},
                "kappa": {"x0": 3, "lbub": [1e-3,5]},
                "theta": {"x0": 0.05, "lbub": [1e-3,0.1]},
                "sigma": {"x0": 0.3, "lbub": [1e-2,1]},
                "rho": {"x0": -0.8, "lbub": [-1,0]},
                "lambd": {"x0": 0.03, "lbub": [-1,1]},
                }
        
        param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho', 'lambd']


        x0 = [param["x0"] for key, param in params.items()]
        bnds = [param["lbub"] for key, param in params.items()]

   
        result = minimize(
            lambda x: Heston.SqErr(x, volSurfaceLong, S0),
            x0,
            method='SLSQP',
            bounds=bnds,
            tol=1e-3,
            options={'maxiter': int(1e4)}
        )

     
     
        Heston.print_optimization_summary(result, param_names, P, price_data= volSurfaceLong["price"])

        return result
    


    def print_optimization_summary(result, param_names, P, price_data=None, title="Résultats de la calibration"):
        print(f"\n=== {title} ===")
        print(f"{'Paramètre':<10} | {'Valeur calibrée':<15}")
        print("-" * 30)
        
        values = [round(val, 4) for val in result.x]
        for name, value in zip(param_names, values):
            print(f"{name:<10} | {value:<15}")

        print("\n--- Statistiques d'erreur ---")
        print(f"Erreur quadratique minimale : {round(result.fun, 4)}")
        print(f"Nombre de points de calibration : {len(P)}")
        print(f"MSE : {round(result.fun / len(P), 6)}")
        print(f"RMSE : {round(np.sqrt(result.fun / len(P)), 6)}")

        if price_data is not None:
            print(f"Moyenne des prix du marché : {round(price_data.mean(), 4)}")

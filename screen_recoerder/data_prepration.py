import json
from pprint import pprint
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px


class DataPreparation(object):
    path_general = f"C:/Users/hayth/Desktop/Projects/Python/Aviator/screen_recoerder/"

    @staticmethod
    def zero_imputation(data):
        for i in data:
            if i['total_bet'] == '':
                i['total_bet'] = '0'
            if i['cash_out'] == '':
                i['cash_out'] = '0'


    @staticmethod
    def missing_bets(data):
        i = 5
        while i < len(data):
            tmp = 0
            step_back = 1
            if data[i]['total_bet'] == '0' or float(data[i]['total_bet']) < 50:
                while step_back < 5:
                    tmp = tmp + int(data[i-step_back]['total_bet'])
                    step_back += 1
                data[i]['total_bet'] = str(tmp // step_back)
            i += 1
        data[2]['total_bet'] = data[1]['total_bet']


    @staticmethod
    def missing_cash_out(data):
        i = 0
        while i < len(data):
            if float(data[i]['cash_out']) == 0:
                # print(data[i])
                j = 0
                test = False
                while j < len(data) and not test:
                    if 0.9 < float(data[j]['result'])/float(data[i]['result']) < 1.1 and float(data[j]['result']) > 1.2:
                        data[i]['cash_out'] = data[j]['cash_out']
                        test = True
                    else:
                        j += 1
                # print(data[i])
                # print("--------------------------")
            i += 1
        for i in data:
            if float(i['result']) > 20 and float(i['cash_out']) == 0:
                i['cash_out'] = str(float(i['total_bet'])-2)
        return 0

    @staticmethod
    def missing_results(data):
        i = 0
        while i < len(data):
            if data[i]['result'] == '' or data[i]['result'][0] == '.':
                data.remove(data[i])
            else:
                i += 1

        return None

    @staticmethod
    def delete_duplicate(data, to_clean):
        i = 1
        while i < len(data):
            if data[i]['result'] == data[i-1]['result']:

                if int(data[i]['total_bet']) < int(data[i-1]['total_bet']):
                    data.remove(data[i])
                elif int(data[i]['total_bet']) >= int(data[i-1]['total_bet']):
                    data.remove(data[i-1])
            else:
                i += 1
        return None

    @staticmethod
    def convention_data(data):
        i = 0
        while i < len(data):
            original_date = datetime.strptime(data[i]['date'], '%Y_%m_%d %H_%M_%S')
            data[i]['date'] = original_date.strftime('%Y-%m-%d %H:%M:%S')
            data[i]['result'] = float(data[i]['result'])
            data[i]['total_bet'] = float(data[i]['total_bet'])
            data[i]['cash_out'] = float(data[i]['cash_out'])
            i += 1
        return None

    @staticmethod
    def exo_endo_parameters(df):
        # df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df_exo = df[['total_bet', 'cash_out']]
        df_endo = df['result']
        return df_endo, df_exo

    @staticmethod
    def augmented_Dickey_Fuller(series, window, visualization=False):
        print(series.describe())
        if visualization:
            # Visualize the time series
            plt.plot(series)
            plt.title('Time Series Data')
            plt.show()

        # Calculate summary statistics
        mean = series.mean()
        variance = series.var()
        print('Mean:', mean)
        print('Variance:', variance)

        # Calculate rolling statistics
        rolling_mean = series.rolling(window=window).mean()
        rolling_variance = series.rolling(window=window).var()
        # pprint(rolling_variance)
        # pprint(rolling_mean)

        if visualization:
            # Plot rolling statistics
            plt.plot(series, label='Original')
            plt.plot(rolling_mean, label='Rolling Mean')
            plt.plot(rolling_variance, label='Rolling Variance')
            plt.title('Rolling Statistics')
            plt.legend()
            plt.show()

        # Perform the ADF test
        result = adfuller(series)

        # Extract the test statistics and p-value
        test_statistic = result[0]
        p_value = result[1]

        print("-----------------------------------------")
        print('test_statistic = ', test_statistic)
        print('p-value = ', p_value)

        stationary = True
        # Compare the p-value to the significance level (e.g., 0.05)
        if p_value < 0.05:
            print("The time series is likely stationary.")
        else:
            print("The time series is likely non-stationary.")
            stationary = False

        return series, stationary

    @staticmethod
    def train_test_split(data, rate):
        split_index = int(rate * len(data))
        train_data = data.iloc[:split_index]  # Take the first 80% of rows as training data
        test_data = data.iloc[split_index:]
        return train_data, test_data

    @staticmethod
    def autocorrelation_analysis(series, visualization=True, plot=True):
        # Convert Series to numpy array
        data = series.values
        if plot:
            # Calculate ACF and PACF
            acf_values = acf(data, nlags=30)
            pacf_values = pacf(data, nlags=30)

            # Print ACF values
            print("ACF:")
            for lag, acf_val in enumerate(acf_values[:25]):
                print(f"Lag {lag + 1}: {acf_val}")
                # Print PACF values
                print("PACF:")
                for lag, pacf_val in enumerate(pacf_values[:25]):
                    print(f"Lag {lag + 1}: {pacf_val}")

        if visualization:
            # Plot Autocorrelation Function (ACF)
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_acf(series, ax=ax, lags=30)  # Adjust the number of lags as needed
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.title('Autocorrelation Plot (ACF)')
            plt.show()
            # Plot Partial Autocorrelation Function (PACF)
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_pacf(series, ax=ax, lags=30)  # Adjust the number of lags as needed
            plt.xlabel('Lag')
            plt.ylabel('Partial Autocorrelation')
            plt.title('Partial Autocorrelation Plot (PACF)')
            plt.show()

    @staticmethod
    def fit_the_ARIMA_model(data, exo_train, exo_test, endo_train, endo_test, p, d, q, value, residuals_plot=True,
                            visualisation=False):
        model = ARIMA(endo_train, exog=exo_train, order=(p, d, q))
        model_fit = model.fit()
        print(model_fit.summary())

        residuals = model_fit.resid[1:]
        if residuals_plot:
            fig, ax = plt.subplots(1, 2)
            residuals.plot(title='Residuals', ax=ax[0])
            residuals.plot(title='Density', kind='kde', ax=ax[1])
            plot_acf(residuals)
            plot_pacf(residuals)

        forecast_test = model_fit.forecast(len(endo_test), exog=exo_test)
        predicted_values = pd.Series([None] * len(endo_train) + list(forecast_test), index=range(len(data)))


        data = pd.DataFrame(data)
        data['predicted_value'] = predicted_values
        fig = px.line(data, x='date', y=[value, 'predicted_value'], title='ARIMAX Model Interpretation')
        fig.show()

        # with open('arima_model.pkl', 'wb') as f:
        #     pickle.dump(model_fit, f)
        return 0



    #---------------------------------------------------------------------------------------------
    def data_loading(self, file_name):
        file_path = self.path_general + file_name
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            print(len(data))
        """ Data Preparation """
        self.zero_imputation(data)
        self.missing_results(data)
        self.delete_duplicate(data, 'total_bet')
        self.missing_bets(data)
        self.missing_cash_out(data)
        self.convention_data(data)

        """ Verification """
        pprint(len(data))
        # pprint(data)
        # i=0
        # while i < len(data):
        #     if data[i]['total_bet'] == '0':
        #         print(i)
        #         print(data[i])
        #         print('---')
        #     i += 1

        """ Time Series """
        df = pd.DataFrame(data)
        train, test = self.train_test_split(df, 0.8)
        endo_train, exo_train = self.exo_endo_parameters(train)
        endo_test, exo_test = self.exo_endo_parameters(test)
        pprint(endo_train)
        pprint(exo_train)
        # series, stationary = self.augmented_Dickey_Fuller(endo_test, 5, visualization=True)
        # self.autocorrelation_analysis(series,visualization=True, plot=False)
        self.fit_the_ARIMA_model(data,exo_train,exo_test,endo_train,endo_test,1,0,1, 'result', True, True )


        return data


re = DataPreparation()
re.data_loading('data_2')

import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import CSV into pandas
csv = pandas.read_csv(r'C:\Users\timmy\PycharmProjects\Life expectancy.csv')

#plot data
g = sns.jointplot(x="Year", y="Life_expectancy", data=csv, kind='reg')
regline = g.ax_joint.get_lines()[0]
regline.set_color('red')
regline.set_zorder(5)
plt.show()

#linear Regression predictor
df = pd.read_csv('Life expectancy.csv')
X = df['Year']
Y = df['Life_expectancy']

#Ask for year input for life expectancy
predict_year = int(input("Enter what year you want the life expectancy for: "))

#regr = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(df.Year, df.Life_expectancy, test_size = 0.2)
regr = LinearRegression()
regr.fit(np.array(x_train).reshape(-1,1), y_train)
predict = regr.predict(np.array(predict_year).reshape(-1,1))

print(predict)

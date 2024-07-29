# y=2x 그래프 그리기
# 점을 직선으로 이어서 표현
x = np.linspace(0, 8, 2) # 0에서 8 범위, 점 2개
y = 2*x
plt.scatter(x, y, s=2, color="red")
plt.plot(x, y)
plt.show()
plt.clf()

# y=x**2
x = np.linspace(-8, 8, 100)
y = x**2
plt.scatter(x, y, s=2, color="red")
plt.plot(x, y)
plt.xlim(-10, 10)
plt.ylim(0, 40)
plt.gca().set_aspect("equal", adjustable="box") # x, y축 간격 똑같이, 박스 형태로
plt.show()
plt.clf()

# 연습문제: 신뢰구간 구하기(평균, n, sigma, a, Za/2)
x=np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean() # 평균: 68.893
len(x) # n:16
# sigma: 6
# 90% 신뢰구간: 1-a = 0.9 -> a=0.1 
# Za/2 = Z0.05
z_005 = norm.ppf(0.95, loc=0, scale=1)
x.mean() + z_005 * 6/np.sqrt(16)
x.mean() - z_005 * 6/np.sqrt(16)

# 데이터로부터 E[X^2] 구하기 X~N(3, 5**2)
x = norm.rvs(loc=3, scale=5, size=10000)
np.mean(x**2)
np.var(x) + (np.mean(x)**2)
# 분산: sum(x**2)/(len(x)-1)
# np.mean((x-x**2)/(2*x))

# 표본분산
np.random.seed(20240729)
x=norm.rvs(loc=3, scale=5, size=100000)
x_bar = x.mean()
s_2 = sum((x-x_bar)**2)/(100000-1) # sum((x-np.mean(x))**2)/(len(x)-1)
np.var(x, ddof=1) # n-1로 나눈 값(표본 분산) # x.var(ddof=1) <- 표본분산 구할 거라면 이게 맞음
np.var(x, ddof=0) # n으로 나눈 값 # x.var() 

x=norm.rvs(loc=3, scale=5, size=20)
np.var(x) # n으로 나눈 값 # 25보다 왼쪽에 있음
np.var(x, ddof=1) # n-1로 나눈 값 # 25에 딱 있음

#=========================================================

#같은 해에 지어진 그룹을 한 그룹으로 보고 -> 평균을 냄
#test.set에 있는 집값을 예측해보자.
house_train = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/train.csv")
house_train=house_train[["Id", "YearBuilt", "SalePrice"]]
house_mean = house_train.groupby("YearBuilt", as_index=False)\
         .agg( 
             mean_year=("SalePrice", "mean")
         )
house_test = pd.read_csv('test.csv')
house_test = house_test[["Id", "YearBuilt"]]

house_test = pd.merge(house_test, house_mean, how = "left", on = "YearBuilt")
house_test = house_test.rename(columns={"mean_year" : "SalePrice"})
house_test["SalePrice"].isna().sum()
house_test.loc[house_test["SalePrice"].isna()]
house_mean = house_test["SalePrice"].mean()
house_test["SalePrice"] = house_test["SalePrice"].fillna(house_mean)

sub_df = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission.csv")
sub_df["SalePrice"] = house_test["SalePrice"]
sub_df.to_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission2.csv", 
                index = False) # index = False 안 하면 열 하나가 추가 됨 

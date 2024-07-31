from scipy.stats import norm

# 하위 25%에 해당하는 x는?
x = norm.ppf(0.25, loc=3, scale=7) # x의 mu, sigma
z = norm.ppf(0.25, loc=0, scale=1)
norm.ppf(0.975, loc=0, scale=1) * 7 + 3

z * 7 + 3
# x = 3 + 7*z
# (x-3)/7 = z
# (x-mu)/sigma = z

norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1)

# 표본정규분포 / 표준(Z) 1000개 뽑고 히스토그램 그리되, pdf 겹치기
z = norm.rvs(loc=0, scale=1, size=1000)
sns.histplot(z, stat="density") # 스케일 맞춰줌

zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100) # 일정한 간격으로 숫자 출력
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color="red", linewidth="2")

plt.show()
plt.clf()

#========================================================

x=z*np.sqrt(2) + 3
sns.histplot(z, stat="density", color="grey")
sns.histplot(x, stat="density", color="green")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()

# 표준화 확인
x=norm.rvs(loc=5, scale=3, size=1000)

# 표준화
z=(x - 5)/3
sns.histplot(z, stat="density", color="grey")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

#========================================================
# 표본표준편차 나눠도 표준정규분포가 되는지 확인 -> 안된다
# 빨간색보다는 밑에 분포가 그려지는 이유는 
x = norm.rvs(loc=5, scale=3, size=10)
s = np.std(x, ddof=1)

x = norm.rvs(loc=5, scale=3, size=1000)

z = (x-5)/s
sns.histplot(z, stat="density", color="grey")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()

# t 분포에 대해 알아보자! X ~ t(df)
# 종모양, 대칭분포, 중심 0
# 모수 df: 자유도라고 부름 - 퍼짐을 나타내는 모수
# df가 작으면 분산이 커짐. df이 무한대로 가면 
from scipy.stats import t

# t.pdf
# t.ppf
# t.cdf
# t.rvs

# 자유도가 4인 t분포의 pdf(정규분포그래프)를 그려보세요!
t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df=4) # df는 자유도(분산은 커지면 퍼지는데 자유도는 커지면 좁아짐)
plt.plot(t_values, pdf_values, color="red", linewidth=2)
# 표준정규분포 그리기
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='green', linewidth=2)

plt.show()
plt.clf()

# X ~ ?(mu, sigma^2)
# X_bar ~ N(mu, sigma^2/n)
# X_bar ~= t(x_bar, s^2/n) <- 자유도가 df-1인 t분포
x = norm.rvs(loc=15, scale=3, size=16, random_state = 42)
n = len(x)
x_bar = x.mean()

# 모분산을 모를 때 모평균에 대한 95% 신뢰구간
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1)/np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1)/np.sqrt(n)

# 모분산(3^2)을 알 때 모평균에 대한 95% 신뢰구간
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3/np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3/np.sqrt(n)

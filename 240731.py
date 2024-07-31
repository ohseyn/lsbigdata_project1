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

#========================================================
# 


#========================================================
# 표본표준편차 나눠도 표준정규분포가 되는지 확인 -> 안된다
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

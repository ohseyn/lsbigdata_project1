house = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/train.csv")
house_mean2=house.groupby(["YearBuilt","GarageCars","KitchenQual"], as_index=False)\
         .agg( 
             house_mean=("SalePrice", "mean")
         )
house_mean2

#test 파일에서 id랑년도만 빼줌
house_test2 = pd.read_csv('test.csv')
house_test2 = house_test2[["Id","YearBuilt","GarageCars","KitchenQual"]]
house_test2

# test 파일에 집값 평균 구한거 연도에 맞게 추가
house_test2 = pd.merge(house_test2, house_mean2, how = "left", on = ["YearBuilt","GarageCars","KitchenQual"])
house_test2

# 열이름 바꿈
house_test2 = house_test2.rename(
    columns = {"house_mean" : "SalePrice"}
)
house_test2

# 결측치 처리
house_test2["SalePrice"].isna().sum()

price_mean = house["SalePrice"].mean()
price_mean

house_test2 = house_test2.fillna(price_mean)

# submission 파일 만듬
submission2 = house_test2[["Id","SalePrice"]]
submission2

submission2.to_csv("sample_submission3.csv", index=False)

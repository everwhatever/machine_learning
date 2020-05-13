
import pandas as pd

sal=pd.read_csv('Salaries.csv')
#print(sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits'])


#print(sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].max()]['EmployeeName'])

#print(sal.groupby('Year')['BasePay'].mean())

#print(sal['JobTitle'].nunique())

#print(sal['JobTitle'].value_counts().head(5))

#print(sum(sal['JobTitle'].value_counts()==1))


# def word(title):
#     if 'chief' in title.lower().split():
#         return True
#     return False
#
# print(sum(sal['JobTitle'].apply(lambda x:word(x))))
ecom=pd.read_csv('Ecommerce.csv')

#print(ecom['Purchase Price'].mean())

#print(ecom['Purchase Price'].min())

#print(sum(ecom['Language']=='en'))

#print(ecom['AM or PM'].value_counts())

#print(ecom['Job'].value_counts().head(5))

#print(ecom[ecom['Lot']=='90 WT']['Purchase Price'])

#print(sum((ecom['CC Provider']=='American Express')&(ecom['Purchase Price']>95)))

#print(sum(ecom['CC Exp Date'].apply(lambda x:x.split('/')[1])=='25'))

#print(ecom['Email'].apply(lambda x:x.split('@')[1]).value_counts().head(5))

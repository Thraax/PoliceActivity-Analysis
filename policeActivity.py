# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

# Import the data set
path = 'F:\\Sprojects\\ML\\PoliceActivity\\police.csv'
ri_data = pd.read_csv(path)

# Looking for nulls

# Summarize null columns by [sum , percentage] of nulls
null_summary = pd.concat([ri_data.isnull().sum(), ri_data.isnull().sum() / ri_data.isnull().count()],
                         axis=1, keys=['Sum', 'Percentage']) \
    .sort_values(ascending=False, by=['Sum'])

# drop the columns with high nulls
ri_data.drop(['county_name', 'search_type'], axis=1, inplace=True)

# Drop rows with null values ==== mostly 5% of 91K+ columns will not affect so much ====
ri_data.dropna(how='any', axis=0, inplace=True)

# Convert columns to correct datatype
'''
Avoid strings as possible and convert them to category if the data contain possible number of output (F/M)
If the data contain contain variable that arrow to event that happened or not (True/False) replace with bool
'''
ri_data.driver_race = ri_data.driver_race.astype('category')
ri_data.driver_gender = ri_data.driver_gender.astype('category')
ri_data.violation_raw = ri_data.violation_raw.astype('category')
ri_data.violation = ri_data.violation.astype('category')
ri_data.search_conducted = ri_data.search_conducted.astype(bool)
ri_data.is_arrested = ri_data.is_arrested.astype(bool)
ri_data.stop_outcome = ri_data.stop_outcome.astype('category')

# Concat the stop time and the date to be the index
combine = ri_data.stop_date.str.cat(ri_data.stop_time, sep=' ')
ri_data['stop_time'] = pd.to_datetime(combine)
ri_data.set_index(ri_data['stop_time'], inplace=True)

#  ===== Does gender affect whose vehicle is searched?  =====
ri_data.groupby(['driver_gender']).search_conducted.sum().sort_values(ascending=False)

#  ===== Does gender affect who is arrested during a search?  =====
search_conducted = ri_data[(ri_data['search_conducted'] == True)]
search_conducted.groupby(['driver_gender']).is_arrested.sum().sort_values(ascending=False)

arrested_barplot = search_conducted.groupby(['driver_gender']).sum().sort_values(ascending=False, by='is_arrested')
arrested_barplot.plot(kind='bar', title='Arrests', ylabel='Arrests',
                      xlabel='Gender', figsize=(6, 5))
plt.xticks(rotation=0)
plt.show()

#  ===== Does race affect who is arrested during a search?  =====
arrested_barplot = search_conducted.groupby(['driver_race']).sum().sort_values(ascending=False, by='is_arrested')
arrested_barplot.plot(kind='bar', title='Arrests', ylabel='Arrests',
                      xlabel='Race', figsize=(6, 5))
plt.xticks(rotation=0)
plt.show()

#    ===== number of arrested in district =====
dist_arrest = ri_data.groupby(['district']).is_arrested.sum().sort_values(ascending=False)
dist_arrest.plot(kind='bar', xlabel='district', ylabel='Arrests', title='Number of arrests by district',
                 figsize=(6, 5))
plt.xticks(rotation=0)
plt.show()

#    Violations by district

vio_dist = pd.crosstab(ri_data.district, ri_data.violation)
vio_dist.plot(kind='bar')
plt.xticks(rotation=0)
plt.show()

#  ===== number of arrested by driver race =====
race_arrest = ri_data.groupby(['driver_race']).is_arrested.sum().sort_values(ascending=False)
race_arrest.plot(kind='bar', xlabel='driver race', ylabel='Arrests', title='Number of arrests by race',
                 figsize=(6, 5))
plt.xticks(rotation=0)
plt.show()

#  Violations by race

vio_dist = pd.crosstab(ri_data.driver_race, ri_data.violation)
vio_dist.plot(kind='bar')
plt.xticks(rotation=0)
plt.show()


# Violations by race and gender

vio_racegender = pd.crosstab(ri_data.driver_race, ri_data.driver_gender).sort_values(ascending=False, by='M')
vio_racegender.plot(kind='bar')
plt.xticks(rotation=0)
plt.show()


# Visualize hourly arrested rate

hourly_arrested = ri_data.groupby(ri_data.index.hour).is_arrested.mean()
hourly_arrested.plot(color='r')
plt.xlabel('Hours')
plt.ylabel('Rate of arrested')
plt.title('Rate of arrest by hours')

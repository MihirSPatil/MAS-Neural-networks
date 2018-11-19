from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


fleisch = pd.read_excel('Archive_files/Archiv/Fleisch.xls').astype(float)
stoff =pd.read_excel('Archive_files/Archiv/Stoff.xls').astype(float)
leder = pd.read_excel('Archive_files/Archiv/Leder.xls').astype(float)
holz = pd.read_excel('Archive_files/Archiv/Holz.xls').astype(float)
haut = pd.read_excel('Archive_files/Archiv/Referenz-Haut_6-Klassen.xls').astype(float)
skin = pd.read_csv('Archive_files/Archiv/2016skin.csv', sep=';', decimal=',').astype(float)
skin.dropna(axis=0, inplace=True)
material = pd.read_csv('Archive_files/Archiv/2016material.csv', sep=';', decimal=',').astype(float)
material.dropna(axis=0, inplace=True)
material_fake = pd.read_csv('Archive_files/Archiv/2016material-fake.csv', sep=';', decimal=',').astype(float)
material_fake.dropna(axis=0, inplace=True)


avg_fleisch = np.array(fleisch.iloc[:,1:].mean(axis = 1))[np.newaxis].T
avg_leder = np.array(leder.iloc[:,1:].mean(axis = 1))[np.newaxis].T
avg_holz = np.array(holz.iloc[:,1:].mean(axis = 1))[np.newaxis].T
avg_stoff = np.array(stoff.iloc[:,1:].mean(axis = 1))[np.newaxis].T
avg_haut = np.array(haut.iloc[:,1:].mean(axis = 1))[np.newaxis].T
avg_skin = np.array(skin.iloc[:,1:].mean(axis = 1))[np.newaxis].T
avg_material = np.array(material.iloc[:,1:].mean(axis = 1))[np.newaxis].T
avg_material_fake = np.array(material_fake.iloc[:,1:].mean(axis = 1))[np.newaxis].T

wavelength = np.array(fleisch['nm'])[np.newaxis].T
wavelength_2016 = np.array(skin['nm'])[np.newaxis].T


from sklearn.decomposition import PCA

def dataCleanup(data):

    gross_avg = np.array(data.iloc[:,1:].mean(axis = 1))

    # print('average of the combined dataset={}'.format(gross_avg.shape))

    mean_removed_data = data.subtract(gross_avg, axis = 0)

#     mean_removed_data.index

    pca = PCA(n_components=5)

    pca.fit(mean_removed_data)

    cleaned_data = pca.transform(mean_removed_data)

    # print('number of components selected to maintain = {}'.format(pca.n_components_))
    # print('the variance ratio for each component = {}'.format(pca.explained_variance_ratio_))

    cleaned_data = pd.DataFrame(index = mean_removed_data.index, data = cleaned_data)
    return cleaned_data


# print('------------- FLEISCH -------------')
fleisch_cleaned = dataCleanup(fleisch.set_index('nm'))
# print('-------------   HAUT  -------------')
haut_cleaned = dataCleanup(haut.set_index('nm'))
# print('-------------   SKIN  -------------')
skin_cleaned = dataCleanup(skin.set_index('nm'))
# print('-----------   MATERIAL  -----------')
material_cleaned = dataCleanup(material.set_index('nm'))
# print('---------   MATERIAL FAKE  --------')
material_fake_cleaned = dataCleanup(material_fake.set_index('nm'))
# print('-------------   HOLZ  -------------')
holz_cleaned = dataCleanup(holz.set_index('nm'))
# print('-------------  LEDER  -------------')
leder_cleaned = dataCleanup(leder.set_index('nm'))
# print('-------------  STOFF  -------------')
stoff_cleaned = dataCleanup(stoff.set_index('nm'))

# print('=========================================================================')
# print("Creating the labels to train the network")
# print('=========================================================================')
fleisch_labels = np.ones((len(fleisch_cleaned),1))
# print("Fleisch Labels: ",fleisch_labels.shape)

haut_labels = np.ones((len(haut_cleaned),1))
# print("Haut Labels: ",haut_labels.shape)

skin_labels = np.ones((len(skin_cleaned),1))
# print("Skin Labels: ",skin_labels.shape)

material_labels = np.zeros((len(material_cleaned),1))
# print("Material Labels: ",material_labels.shape)

material_fake_labels = np.zeros((len(material_fake_cleaned),1))
# print("Material Fake Labels: ",material_fake_labels.shape)

holz_labels = np.zeros((len(holz_cleaned),1))
# print("Holz Labels: ",holz_labels.shape)

leder_labels = np.zeros((len(leder_cleaned),1))
# print("Leder Labels: ",leder_labels.shape)

stoff_labels = np.zeros((len(stoff_cleaned),1))
# print("Stoff Labels: ", stoff_labels.shape)

# print('=========================================================================')
# print("Combining all the datasets and labels")
# print('=========================================================================')
ultimate_dataset = np.vstack((fleisch_cleaned, holz_cleaned, leder_cleaned, haut_cleaned,\
                             stoff_cleaned, skin_cleaned, material_cleaned, material_fake_cleaned))
# print("Dataset size: ", ultimate_dataset.shape)
ultimate_labels = np.vstack((fleisch_labels, holz_labels, leder_labels, haut_labels,\
                            stoff_labels, skin_labels, material_labels, material_fake_labels))
# print("Dataset labels size: ", ultimate_labels.shape)


X_train, X_test, y_train, y_test = train_test_split(ultimate_dataset, ultimate_labels, test_size = 0.20)

svclassifier = SVC(kernel='poly')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

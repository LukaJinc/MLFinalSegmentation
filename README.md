# MLFinalSegmentation

ამ Repo-ში აღწერილია პოპულაციის სეგმენტაციის tool ის დეველოპმენტი და დაწერილია საბოლოო მეთოდები რომელთა საშუალებითაც კლიენტს შეეძლება ნებისმიერი პოპულაციის ნებისმიერი ცვლადებით სეგმენტაცია, ასევე პროექტის მნიშვნელოვანი კომპონენტია ის, რომ მიღებულ სეგმენტების ახსნასაც ეს tool დააგენერირებს მოცემული ცვლადებით. ეს tool გატესტილია რამდენიმე dataset ზე. ასევე შემოწმებულია რამდენიმე სხვადასხვა მიდგომით ამ tool-ის მდგრადობა.

პირველადი გეგმა არის შემდეგი:
1. მოცემული ცვლადები აუცილებელია შევკუმშოთ ცოტა ცვლადზე, რადგან სეგმენტების დასაგენერირებლად ვაპირებ KMeans ალგორითმის გამოყენებას, ხოლო მისთვის არსებითად მნიშვნელოვანია ცოტა ცვლადის გამოყენება, რადგან მაღალ განზომილებაში distance ის ცნებას კარგავს და შესაბამისად არსობრივ სეგმენტებს ვეღარ ვიღებთ. ანუ უნდა გამოვიყენოთ PCA. თუ ცვლადების რაოდენობა მეტია 20 ზე, n_components=20 ხოლო თუ ათიდან ოცამდეა n_components=10 რადგან pca ით კატეგორიულების პრობლემასაც ვაგვარებთ(k means clustering ში პირდაპირ ბინარული ცვლადის შეშვება არაა სწორი რადგან სხვა განაწილება აქვს და დისტანციის დათვლის ცნებას აფუჭებს) და ამ ნაწილში იქნება გამოყენების მხრივ ერთადერთი შეზღუდვა - PCA Result >= 70%. ეს გვჭირდება იმისათვის, რომ უზრუნველყოთ კლიენტის მხრიდან მოწოდებულ ცვლადებში არსებული ინფორმაცია რომ PCA ის მიერ გამოყვანილ ცვლადებშიც აისახოს. ანუ გამოყენების მხრივ კლიენტის მხარეს რჩება ის პასუხისმგებლობა, რომ რელევანტური და "კარგი" ცვლადები მოგვცეს. Garbage In, Garbage Out
2. Outliers ის მოშორება. ამ ეტაპზე ამისათვის გამოვიყენებ interquantile range - ს და ამას გავაკეთებ pca ის შემდეგ.
3. Elbow Method/Silhouette Score Analysis - ოპტიმალური კლასტერების რაოდენობის შერჩევა
4. Cluster Description - Decision Tree სტრუქტურის გამოყენებით თითოეული სეგმენტის Description ის დაგენერირება

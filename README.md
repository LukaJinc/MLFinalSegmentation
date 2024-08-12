# MLFinalSegmentation

ამ Repo-ში აღწერილია პოპულაციის სეგმენტაციის tool ის დეველოპმენტი და დაწერილია საბოლოო მეთოდები რომელთა საშუალებითაც კლიენტს შეეძლება ნებისმიერი პოპულაციის ნებისმიერი ცვლადებით სეგმენტაცია, ასევე პროექტის მნიშვნელოვანი კომპონენტია ის, რომ მიღებულ სეგმენტების ახსნასაც ეს tool დააგენერირებს მოცემული ცვლადებით.

პროექტის გეგმა არის შემდეგი:
1. მოცემული ცვლადები აუცილებელია შევკუმშოთ ცოტა ცვლადზე, რადგან სეგმენტების დასაგენერირებლად ვაპირებ KMeans ალგორითმის გამოყენებას, ხოლო მისთვის არსებითად მნიშვნელოვანია ცოტა ცვლადის გამოყენება, რადგან მაღალ განზომილებაში distance ის ცნებას კარგავს და შესაბამისად არსობრივ სეგმენტებს ვეღარ ვიღებთ. ანუ უნდა გამოვიყენოთ PCA.

თუ ცვლადების რაოდენობა მეტია 20 ზე, n_components=20 ხოლო თუ ათიდან ოცამდეა n_components=10 რადგან pca ით კატეგორიულების პრობლემასაც ვაგვარებთ(k means clustering ში პირდაპირ ბინარული ცვლადის შეშვება არაა სწორი რადგან სხვა განაწილება აქვს და დისტანციის დათვლის ცნებას აფუჭებს) და ამ ნაწილში იქნება გამოყენების მხრივ ერთადერთი შეზღუდვა - PCA Result >= 70%. 

ეს გვჭირდება იმისათვის, რომ უზრუნველყოთ კლიენტის მხრიდან მოწოდებულ ცვლადებში არსებული ინფორმაცია რომ PCA ის მიერ გამოყვანილ ცვლადებშიც აისახოს. ანუ გამოყენების მხრივ კლიენტის მხარეს რჩება ის პასუხისმგებლობა, რომ რელევანტური და "კარგი" ცვლადები მოგვცეს. Garbage In, Garbage Out

2. Outliers ის მოშორება. ამ ეტაპზე ამისათვის გამოვიყენებ interquantile range - ს და ამას გავაკეთებ pca ის შემდეგ.

3. K-Means ალგორითმის გამოყენებით სხვადასხვა კლასტერების რაოდენობების სეგმენტების გენერაცია.
   
4. Cluster Description - Decision Tree სტრუქტურის გამოყენებით თითოეული სეგმენტის Description ის დაგენერირება. მნიშვნელოვანი მომენტია ის, რომ max_depth=3, რათა მიღებული აღწერები არ იყოს ძალიან ჩახლართული. 

მაგალითად მოცემული გვაქვს სამი კლასტერი და გვინდა მათი აღწერების გენერაცია. თითოეული სეგმენტისთვის დავატრენინგებთ ცალკე Decision Tree - ს, ამოვხსნით ბინარული კლასიფიკაციის ამოცანას, რომელიც ერთმანეთისგან არჩევს მიმდინარე კლასტერს დანარჩენებისაგან. 

შემდგომ ავიღებთ Leaf, რომელიც ეკუთვნის მიმდინარე კლასს და ამავდროულად იჭერს კლასტერიდან ყველაზე მეტ წერტილს. მიმდინარე კლასტერის აღწერა კი იქნება root დან leaf მდე ნოუდებში ჩაწერილი feature ბი და შესაბამისი threshold ები.

ასევე გავტესტე MultiClass Decision Tree. განსხვავება ის არის ზედა მიდგომისგან, რომ K კლასტერის შემთხვევაში ვატრენინგებთ ერთ ხეს და არა K-ს. MultiClass Tree - ში ერთდროულად არის ყველა კლასტერის აღწერა. აქაც, თუ ერთ კლასს ორი ფოთოლი შეესაბამება, ვიღებთ რომელიც უფრო მეტ წერტილს ახასიათებს ამ კლასტერიდან.

5. Evaluation - ამ Tool ის შეფასებაში არის ორი მთავარი გამოწვევა:

ა) Custom Metrics - კონკრეტული დაკლასტერების ადეკვატურად შეფასება. იმისათვის, რომ შევადაროთ რა რაოდენობის კლასტერებია ოპტიმალური, გვჭირდება თითოეულის რაღაც მეტრიკით შეფასება. პრობლემა არის ის, რომ კლასიკური Unsupervised Learning ის მეთოდები აქ არ გამოდგება, რადგან ჩვენთვის მნიშვნელოვანია ახსნადი სეგმენტების გენერაცია (4 გეომეტრიულად სწორ სეგმენტს გვირჩევნია 3, რომლებიც 2-3 feature ის მეშვეობით მაღალი სიზუსტით ხასიათდება). 

ყოველი data point - ისათვის:
label - K-Means ით მინიჭებული კლასტერის ნომერი
prediction - ზემოთ მოცემული პროცესის შემდგომ გვექნება თითოეული კლასტერის შესაბამისი select. აქედან გამომდინარე, რომელ select საც დააკმაყოფილებს კონკრეტული data point, იმ კლასტერის ნომერი იქნება მისი prediction

ამ ორი განმარტებიდან კი ჩვენ უკვე შეგვიძლია Supervised Learning იდან Classification ის კლასიკური მეტრიკები გამოვიყენოთ შესაფასებლად. მე ყველაზე მეტად მომწონს f1-score, რადგან precission, recall ჩვენთვის ორივე თანაბრად მნიშვნელოვანია და იდეურად თანხვედრაში მოდის ჩვენ მთავარ გამოწვევასთან:
Precision - მიღებული select რამდენად ზუსტია
Recall - რამდენად აიხსნება კარგად K-Means ის დაგენერირებული კლასტერი მოცემული feature ბით.

ჩვენს იმპლემენტაციაში ასევე არის roc_auc_score, თუმცა იქიდან გამომდინარე, რომ დაუბალანსებელი დატა გამოდის უმეტეს შემთხვევაში, roc_auc_score მაღალია. 

ბ) Overfit - გვჭირდება მექანიზმი, რომლითაც გავაკონტროლებთ იმას, რომ ზედმეტად არ ჩავშალოთ სეგმენტები და განზოგადებადი და მდგრადი იყოს ჩვენ მიერ მოცემული პოპულაციის დახასიათება. ამისათვის ML-ში ფართოდ გავრცელებული Train/Test ად დაყოფა გამოვიყენეთ. 

დავყოთ პოპულაცია Train/Test ად. K-Means დავატრენინგოთ Train ზე და Decision Tree ებით დავაგენერიროთ აღწერები. მიღებული ცენტროიდებითა, description ებითა და ზემოთ აღწერილი შეფასების მექანიზმით შევაფასოთ test. 

შევხედოთ განსხვავებას train/test შორის. ამ მექანიზმით ჩვენ შევძლებთ overfit ის კონტროლს

## rPPG BASED HEART RATE ESTIMATION USING DEEP LEARNING
In this project, we used [MTTS-CAN](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf "MTTS-CAN") toolbox for implementations of deep based methods.
                             
#### ABSTRACT
Remote heart rate estimation is the measurement of heart rate without any physical contact with the patients. This is accomplished using remote photoplethysmography (rPPG). rPPG signals are usually collected using a video camera with a limitation of being sensitive to multiple contributing factors, such as different skin tones, lighting condition of environment and facial structure. There are multiple studies and generally two basic approaches in the literature to process and make sense of these signals: Firstly, we examined the traditional methods as CHROM DEHAN [1], ICA POH [2], GREEN VERCRUYSSE [3] and POS WANG [4]. Secondly, we examined MTTS-CAN [5], one of the deep learning methods. While we tried traditional methods with the UBFC [6] dataset, we ran deep learning methods with UBFC and PURE [7] datasets. When we used SNR [8]  to calculate heart rate based on the Blood Volume Pulse (BVM) signal resulting from deep learning-based methods, we observed a significant improvement in some results. In summary, we concluded that deep learning-based methods play an important role in the development of rPPG technologies and their introduction into our daily lives.

#### METHOD
<img src="https://github.com/esra-polat/rPPG-heart-rate-estimation-deep-learning-method/blob/main/documents/img/mttscan.png" width="800">
We planned to carry out our measurements with deep learning methods, which was our main approach. We hoped that deep learning reduced error rates as a result of these measurements. We used the model of MTTS-CAN to obtain the heart rate signals. This method processes RGB values captured by cameras with functions that also contain certain calculations for various external factors. These external factors include non-physiological variations such as the flickering of the light source, head rotation, and facial expressions. In this method, there are Temporal Shift Modules that will facilitate the exchange of information between frames. These modules provide superior performance in both latency and accuracy. MTTS-CAN also calculates the respiratory rate along with the heartbeat. Since respiration and pulse frequencies cause head and chest movements of the body, calculating these two values together had a great impact on the accuracy of the values compared to independently calculated models. [5]

#### ARCHITECTURE
Flowchart of the proposed algorithm
<img src="https://github.com/esra-polat/rPPG-heart-rate-estimation-deep-learning-method/blob/main/documents/img/flowchart.png" width="800">

#### RESULTS
In the table, we could see the results of traditional methods which are CHROM DEHAN, ICA POH, GREEN VERCRUYSSE and POS WANG methods. In a deep learning-based method which is MTTS-CAN. For example, if we look at 17.avi for all methods, we calculated that deep learning has dropped below five. This result is quite good for us. When we look at the average RMSE values of these methods, we see that the deep learning-based method gives the best results because it has the lowest RMSE.
<img src="https://github.com/esra-polat/rPPG-heart-rate-estimation-deep-learning-method/blob/main/documents/img/res.png" width="700">

#### CONCLUSION AND FUTURE WORK
We worked with the traditional methods in the first term and worked with deep-based methods in the second term. According to the information from literature studies and our studies throughout the year, we can say that deep learning-based methods generally give more correct and faster results than traditional methods. In addition,  when we used SNR to calculate heart rate based on the Blood Volume Pulse (BVP) signal resulting from deep learning-based methods, we observed a significant improvement in some results. As a result, we can say that deep learning-based methods play an important role in the development of rPPG technologies and their introduction into our daily lives.
In the pandemic period, telehealth and remote health monitoring have become increasingly important and people widely expect that this will have a permanent effect on healthcare systems. These tools can help reduce the risk of discovering patients and medical staff to infection, make healthcare services more reachable, and allow doctors to see more patients. In this context, we believe that it will find a place both in health centres and in all kinds of electronic devices. As we can see from the technology news that comes out every day, leading universities of education and leading companies in technology have also concentrated on rPPG studies and both contribute to the literature with research to solve the problems in rPPG or develop new methods. In the next few years, it seems quite possible to open the front camera of our mobile phone and measure our heart rate while sitting at home. Of course, there is no limit to the number of applications to which this technology will be integrated.

-------------
The project is developed by
* Esra POLAT - https://github.com/esra-polat
* Nur Deniz ÇAYLI - https://github.com/nurdenizcayli
* Minel SAYGISEVER - https://github.com/minelsaygisever    

Supervised by Prof. Dr. Çiğdem EROĞLU ERDEM

And this project was awarded the third best project. 
You can find all the details in our [thesis](https://github.com/esra-polat/rPPG-heart-rate-estimation-deep-learning-method/blob/main/documents/Thesis.pdf "thesis").  
<img src="https://github.com/esra-polat/rPPG-heart-rate-estimation-deep-learning-method/blob/main/documents/img/plaket.jpeg" width="1000">

> You can go to the first phase of the project from this [link](https://github.com/esra-polat/rPPG-heart-rate-estimation-traditional-method "link").        
---------------
###### [1] De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886  
###### [2] Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010) Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774 
###### [3] Vercruysse, W., Svasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445. 
###### [4] W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, “Algorithmic principles of remote ppg,” IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, pp. 1479–1491, 2016 
###### [5] Xin Liu, Josh Fromm, Shwetak Patel, Daniel McDuff, “Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement”, NeurIPS 2020, Oral Presentation (105 out of 9454 submissions) 
###### [6] S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, (2017), Unsupervised skin tissue segmentation for remote photoplethysmography, Pattern Recognition Letters 
###### [7] Stricker, R., Müller, S., Gross, H.-M. “Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot” in Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
###### [8] Remote Photoplethysmography Using Nonlinear Mode Decomposition, Halil Demirezen, Cigdem Eroglu Erdem Marmara University Department of Computer Engineering, Goztepe, Istanbul, Turkey, pp. 1060– 1064, 2018.


#### POSTER
            
<img src="https://github.com/esra-polat/rPPG-heart-rate-estimation-deep-learning-method/blob/main/documents/img/poster.jpg">

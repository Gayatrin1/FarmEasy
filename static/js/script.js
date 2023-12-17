function reset( )
{
	document.getElementsByName("N")[0].innerHTML="";
	document.getElementsByName("P")[0].innerHTML="";
	document.getElementsByName("K")[0].innerHTML="";
	document.getElementsByName("temperature")[0].innerHTML="";
	document.getElementsByName("humidity")[0].innerHTML="";
	document.getElementsByName("ph")[0].innerHTML="";
	document.getElementsByName("rainfall")[0].innerHTML="";
}
function disp(s)
{ 
  document.getElementById('reco').innerHTML="<strong>"+s.toUpperCase()+"</strong>";
  document.getElementById('fit').src = "..\\static\\images\\"+s+".jpg";
  document.getElementById('graph').src = "..\\static\\images\\"+s+"G"+".jpg";
  document.getElementById('graph').style.visibility = 'visible';

  let N, P, K, temperature, humidity, ph, rainfall;
  if(s=='apple')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 120 - 45</p>"
   K = "<p>K: 195 - 205</p>"
   Temperature = "<p>Temperature: 21.0365 - 23.9969</p>"
   Humidity = "<p>Humidity: 90.0258 - 94.9205</p>"
   pH = "<p>pH: 5.51425 - 6.49923</p>"
  Rainfall  = "<p>Rainfall: 100.117 - 124.983</p>"
  }
  else if(s=='banana')
  {
   N = "<p>N: 80 - 120</p>" 
   P = "<p>P: 70 - 95</p>"
   K = "<p>K: 45 - 55</p>"
   Temperature = "<p>Temperature: 25.0102 - 29.9089</p>"
   Humidity = "<p>Humidity: 75.0319 - 84.9785</p>"
   pH = "<p>pH: 5.50539 - 6.49007</p>"
  Rainfall  = "<p>Rainfall: 90.1098 - 119.848</p>"

  }
  else if(s=='blackgram')
  {
   N = "<p>N: 20 - 60</p>" 
   P = "<p>P: 55 - 80</p>"
   K = "<p>K: 15 - 25</p>"
   Temperature = "<p>Temperature: 25.0974 - 34.9466</p>"
   Humidity = "<p>Humidity: 60.0653 - 69.961</p>"
   pH = "<p>pH: 6.50014 - 7.77531</p>"
  Rainfall  = "<p>Rainfall: 60.4179 - 74.9156</p>"

  }
  else if(s=='chickpea')
  {
   N = "<p>N: 20 - 60</p>" 
   P = "<p>P: 55 - 80</p>"
   K = "<p>K: 75 - 85</p>"
   Temperature = "<p>Temperature: 17.025 - 20.995</p>"
   Humidity = "<p>Humidity: 14.258 - 19.9698</p>"
   pH = "<p>pH: 5.98899 - 8.86874</p>"
  Rainfall  = "<p>Rainfall: 65.1137 - 94.7819</p>"

  }
  else if(s=='coconut')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 5 - 30</p>"
   K = "<p>K: 25 - 35</p>"
   Temperature = "<p>Temperature: 25.0087 - 29.8691</p>"
   Humidity = "<p>Humidity: 90.0173 - 99.9819</p>"
   pH = "<p>pH:5.50158 - 6.47047</p>"
  Rainfall  = "<p>Rainfall: 131.09 - 225.632</p>"

  }
  else if(s=='coffee')
  {
    N = "<p>N: 80 - 120</p>" 
   P = "<p>P: 15 - 40</p>"
   K = "<p>K: 25 - 35</p>"
   Temperature = "<p>Temperature: 23.0595 - 27.9237</p>"
   Humidity = "<p>Humidity: 50.0456 - 69.9481</p>"
   pH = "<p>pH: 6.02095 - 7.49319</p>"
  Rainfall  = "<p>Rainfall: 115.156 - 199.474</p>"

  }
  else if(s=='cotton')
  {
   N = "<p>N: 100 - 140</p>" 
   P = "<p>P: 35 - 60</p>"
   K = "<p>K: 15 - 25</p>"
   Temperature = "<p>Temperature:22.0009 -25.9924</p>"
   Humidity = "<p>Humidity: 75.0054 - 84.8767</p>"
   pH = "<p>pH: 5.80105 - 7.99468</p>"
  Rainfall  = "<p>Rainfall: 60.6538 - 99.931</p>"

  }
  else if(s=='grapes')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 120 - 145</p>"
   K = "<p>K: 195 - 205</p>"
   Temperature = "<p>Temperature: 8.82567 - 41.9487</p>"
   Humidity = "<p>Humidity: 80.0164 - 83.9835</p>"
   pH = "<p>pH: 5.51092 - 6.4996</p>"
  Rainfall  = "<p>Rainfall: 65.011 - 74.9151</p>"
  }
  else if(s=='jute')
  {
   N = "<p>N: 60 - 100</p>" 
   P = "<p>P: 35 - 60</p>"
   K = "<p>K: 35 - 45</p>"
   Temperature = "<p>Temperature: 23.0943 - 26.9858</p>"
   Humidity = "<p>Humidity: 70.8826 - 89.8911</p>"
   pH = "<p>pH: 6.00252 - 7.48801</p>"
  Rainfall  = "<p>Rainfall: 150.236 - 199.836</p>"

  	
  }
  else if(s=='kidneybeans')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 55 -  80</p>"
   K = "<p>K: 15 - 25</p>"
   Temperature = "<p>Temperature: 15.3304 - 24.9236</p>"
   Humidity = "<p>Humidity: 18.0922 - 24.9697</p>"
   pH = "<p>pH: 5.503 - 5.99812</p>"
  Rainfall  = "<p>Rainfall: 60.2755 - 149.744</p>"

  }
  else if(s=='lentil')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 55- 80</p>"
   K = "<p>K: 15 - 25</p>"
   Temperature = "<p>Temperature: 18.0649 - 29.9441</p>"
   Humidity = "<p>Humidity: 60.0912- 69.9238</p>"
   pH = "<p>pH:  5.91645- 7.8415</p>"
  Rainfall  = "<p>Rainfall: 35.0348 - 54.9394</p>"

  }
  else if(s=='maize')
  {
    N = "<p>N: 60 - 100</p>" 
   P = "<p>P: 35 - 60 </p>"
   K = "<p>K: 15 - 25</p>"
   Temperature = "<p>Temperature: 18.0419 - 26.5499</p>"
   Humidity = "<p>Humidity: 55.2822 - 74.8291</p>"
   pH = "<p>pH: 5.5137 - 6.99584</p>"
  Rainfall  = "<p>Rainfall: 60.6517 - 109.752</p>"

  }
  else if(s=='mango')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 15 - 40</p>"
   K = "<p>K: 25 - 35</p>"
   Temperature = "<p>Temperature: 27.0032 - 35.9901</p>"
   Humidity = "<p>Humidity: 45.0224 - 54.96421</p>"
   pH = "<p>pH: 4.50752 - 6.96742</p>"
  Rainfall  = "<p>Rainfall: 89.2915 - 100.812</p>"

  }
  else if(s=='mothbeans')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 35 - 60</p>"
   K = "<p>K: 15 - 25</p>"
   Temperature = "<p>Temperature: 24.0813 - 31.9993</p>"
   Humidity = "<p>Humidity: 40.0093- 64.9559</p>"
   pH = "<p>pH: 3.50475 - 9.93509</p>"
  Rainfall  = "<p>Rainfall: 30.9201- 74.4433</p>"

  }
  else if(s=='mungbean')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 35 - 60 </p>"
   K = "<p>K: 15 - 25 </p>"
   Temperature = "<p>Temperature: 27.0147- 29.9145</p>"
   Humidity = "<p>Humidity: 80.035 - 89.9962</p>"
   pH = "<p>pH: 6.21892 - 7.1995</p>"
  Rainfall  = "<p>Rainfall: 36.1204 - 59.8723</p>"

  }
  else if(s=='muskmelon')
  {
   N = "<p>N: 80 - 120 </p>" 
   P = "<p>P:  5 -30 </p>"
   K = "<p>K: 45 - 55</p>"
   Temperature = "<p>Temperature: 27.0242 - 29.9435</p>"
   Humidity = "<p>Humidity: 90.0151 - 94.9622</p>"
   pH = "<p>pH: 6.00293 - 6.78105</p>"
  Rainfall  = "<p>Rainfall: 20.2113 - 29.8668</p>"

  }
  else if(s=='orange')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 5 - 30 </p>"
   K = "<p>K: 5 - 15</p>"
   Temperature = "<p>Temperature: 10.0108 - 34.9067</p>"
   Humidity = "<p>Humidity: 90.0062 - 94.9642</p>"
   pH = "<p>pH:6.01039- 7.99585 </p>"
  Rainfall  = "<p>Rainfall: 100.174 - 119.695</p>"

  }
  else if(s=='papaya')
  {
   N = "<p>N:31 - 70 </p>" 
   P = "<p>P: 46 - 70 </p>"
   K = "<p>K:  4 - 55 </p>"
   Temperature = "<p>Temperature: 23.0124 - 43.6755 </p>"
   Humidity = "<p>Humidity:90.0386 - 94.9448</p>"
   pH = "<p>pH: 6.50152 - 6.99347</p>"
  Rainfall  = "<p>Rainfall:40.3515 - 248.859</p>"

  }
  else if(s=='pigeonpeas')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 120 - 45</p>"
   K = "<p>K: 195 - 205</p>"
   Temperature = "<p>Temperature: 0 - 40</p>"
   Humidity = "<p>Humidity: 0 - 40</p>"
   pH = "<p>pH: 0 - 40</p>"
  Rainfall  = "<p>Rainfall: 0 - 40</p>"

  }
  else if(s=='pomegranate')
  {
   N = "<p>N: 0 - 40</p>" 
   P = "<p>P: 5 - 30 </p>"
   K = "<p>K: 35 - 45</p>"
   Temperature = "<p>Temperature: 18.0713 - 24.9627</p>"
   Humidity = "<p>Humidity: 85.1291 - 94.999</p>"
   pH = "<p>pH: 5.56185 - 7.1995 </p>"
  Rainfall  = "<p>Rainfall: 102.518 -112.475</p>"

  }
  else if(s=='rice')
  {
   N = "<p>N:60 - 99</p>" 
   P = "<p>P: 35 - 60</p>"
   K = "<p>K:35 - 45</p>"
   Temperature = "<p>Temperature: 20.0454 - 26.93</p>"
   Humidity = "<p>Humidity:80.1277 - 84.9691</p>"
   pH = "<p>pH: 5.00531 - 7.86847</p>"
  Rainfall  = "<p>Rainfall: 182.562 - 298.56</p>"

  }
  else if(s=='watermelon')
  {
   N = "<p>N: 80 - 120</p>" 
   P = "<p>P:05 - 30</p>"
   K = "<p>K:45 - 55</p>"
   Temperature = "<p>Temperature: 24.0436 - 26.986</p>"
   Humidity = "<p>Humidity:80.0262 - 89.9841</p>"
   pH = "<p>pH:6.00098 - 6.95651</p>"
  Rainfall  = "<p>Rainfall:40.1265 - 59.7598</p>"

  }

 // document.getElementById("why").style.borderRadius = "0.05rem";
 // document.getElementById("why").style.border = "0.25rem";
 document.getElementById("why").style.border = "thick solid #000000";
 document.getElementById("why").style.borderRadius = "1rem";
 document.getElementById('why').innerHTML="<strong>"+N+P+K+Temperature+Humidity+pH+Rainfall+"</strong>";
 document.getElementById('why').style.backgroundColor = "green";
 document.getElementById('why').style.color = "#FFFDD0";
 document.getElementById("why").style.marginTop = "1rem";
 document.getElementById("why").style.width = "30rem";
 document.getElementById("why").style.padding = "2rem";

 
}
function x( )
{
	document.getElementById('reco').innerHTML="Kidney Beans"
	document.getElementById('fit').src = "..\\static\\images\\rajma.jpg";

}
  function gotPosition(pos) 
{
    let lat = pos.coords.latitude;
    let long = pos.coords.longitude;
    console.log(lat);
    console.log(long);
    getForecast(lat, long);
    getAdd(lat, long);
}
function getAdd(lat, long) 
{
   console.log(lat, long);
   // https://maps.googleapis.com/maps/api/geocode/json?latlng=40.714224,-73.961452&key=YOUR_API_KEY   
   //https://geokeo.com/geocode/v1/reverse.php?lat=40.74842&lng=-73.9856&api=YOUR_API_KEY
   // https://api.mapbox.com/geocoding/v5/mapbox.places/-73.62107402003166,40.70763290213338.json?country=in&types=postcode%2Cdistrict%2Cplace%2Cregion%2Ccountry&limit=1&access_token=YOUR_MAPBOX_ACCESS_TOKEN
   let latlng = lat+","+long;   
   let YOUR_API_KEY="18201aa2617fd43ecf0698990f07dda4";
   // let url = "https://maps.googleapis.com/maps/api/geocode/json?latlng="+latlng+"&key="+YOUR_API_KEY;
   // let url = "https://geokeo.com/geocode/v1/reverse.php?lat="+lat+"&lng="+long+"&api="+YOUR_API_KEY;
   // let url = "https://api.mapbox.com/geocoding/v5/mapbox.places/"+latlng+".json?country=in&types=postcode%2Cdistrict%2Cplace%2Cregion%2Ccountry&limit=1&access_token="+YOUR_API_KEY;
   let url= "http://api.positionstack.com/v1/reverse?access_key=18201aa2617fd43ecf0698990f07dda4&query="+latlng;
   getAddText(url); 
}
function getForecast(lat, long) 
{
   let url = "https://api.openweathermap.org/data/2.5/onecall?lat=" + lat + "&lon=" + long + "&exclude=current,minutely,hourly&appid=6c94be434f986b412543b262a76f8125";
   getWeatherText(url);
        
}
async function getAddText(url) 
{
   let addObject = await fetch(url);
   let addText = await addObject.text();
   console.log(addObject);
   console.log(addText);
   parseAdd(addText);       
}  
async function getWeatherText(url) 
{
   let weatherObject = await fetch(url);
   let weatherText = await weatherObject.text();
   console.log(weatherObject);
   console.log(weatherText);
   parseWeather(weatherText);       
}
function parseWeather(weatherText)
{
 let weatherJSON = JSON.parse(weatherText);
 console.log(weatherJSON);
 let humidity = weatherJSON.daily[0].humidity;
 let avgTemp = (kToC(weatherJSON.daily[0].temp.min) + kToC(weatherJSON.daily[0].temp.max))/2;
 document.getElementById('humidity').value = humidity;
 document.getElementById('temperature').value = avgTemp;
}
function parseAdd(addText)
{
 let addJSON = JSON.parse(addText);
 console.log(addJSON);
    let data = addJSON["data"];
    for(var i=0; i<Object.keys(data).length; i++)
    {
      if(data[i].county!=null)
      {
        data = data[i];
        break;
      }
      
    }
 let district="";
 console.log(data);
 if(data!="")
 {
  district= data.county;
  console.log(district);
 }
 for(var i=0; i<Object.keys(dist).length; i++)
{
  if(dist[i].District==district)
  {
    // alert(district);
    record=dist[i];
    break;
  }
}
if(record)
{
  console.log("Hello Record");
  if(range)
{
  console.log("Hi!");
  console.log(range);
}
  PhS = record["Ph Stats"];
  NS = record["N Stats"];
  PS = record["P Stats"];
  KS = record["K Stats"];
      if(PhS!="NA")
      {
        document.getElementById("ph").value=range.Ph[PhS].value;
      }
      if(NS!="NA")
      {
        document.getElementById("N").value=range.N[NS].value;
      }
      if(PS!="NA")
      {
        document.getElementById("P").value=range.P[PS].value;
      }
      if(KS!="NA")
      {
        document.getElementById("K").value=range.K[KS].value;
      }
    
}

//  document.getElementById('humidity').innerHTML = humidity;
//  document.getElementById('temperature').innerHTML = avgTemp;
}
// let kToC = function(kelvinTemp) {
//             const celsius = Math.floor(kelvinTemp - 273);
//             // const fahrenheit = Math.floor(celsius * (9 / 5) + 32);
//             // return fahrenheit
//             return celsius;
//         }
function trigger( )
  { 
    console.log("Triggered");
    navigator.geolocation.getCurrentPosition(gotPosition);
    event.preventDefault();
  } function gotPosition(pos) 
{
    let lat = pos.coords.latitude;
    let long = pos.coords.longitude;
    console.log(lat);
    console.log(long);
    getForecast(lat, long);
    getAdd(lat, long);
}
function getAdd(lat, long) 
{
   console.log(lat, long);
   // https://maps.googleapis.com/maps/api/geocode/json?latlng=40.714224,-73.961452&key=YOUR_API_KEY   
   //https://geokeo.com/geocode/v1/reverse.php?lat=40.74842&lng=-73.9856&api=YOUR_API_KEY
   // https://api.mapbox.com/geocoding/v5/mapbox.places/-73.62107402003166,40.70763290213338.json?country=in&types=postcode%2Cdistrict%2Cplace%2Cregion%2Ccountry&limit=1&access_token=YOUR_MAPBOX_ACCESS_TOKEN
   let latlng = lat+","+long;   
   let YOUR_API_KEY="18201aa2617fd43ecf0698990f07dda4";
   // let url = "https://maps.googleapis.com/maps/api/geocode/json?latlng="+latlng+"&key="+YOUR_API_KEY;
   // let url = "https://geokeo.com/geocode/v1/reverse.php?lat="+lat+"&lng="+long+"&api="+YOUR_API_KEY;
   // let url = "https://api.mapbox.com/geocoding/v5/mapbox.places/"+latlng+".json?country=in&types=postcode%2Cdistrict%2Cplace%2Cregion%2Ccountry&limit=1&access_token="+YOUR_API_KEY;
   let url= "http://api.positionstack.com/v1/reverse?access_key=18201aa2617fd43ecf0698990f07dda4&query="+latlng;
   getAddText(url); 
}
function getForecast(lat, long) 
{
   let url = "https://api.openweathermap.org/data/2.5/onecall?lat=" + lat + "&lon=" + long + "&exclude=current,minutely,hourly&appid=6c94be434f986b412543b262a76f8125";
   getWeatherText(url);
        
}
async function getAddText(url) 
{
   let addObject = await fetch(url);
   let addText = await addObject.text();
   console.log(addObject);
   console.log(addText);
   parseAdd(addText);       
}  
async function getWeatherText(url) 
{
   let weatherObject = await fetch(url);
   let weatherText = await weatherObject.text();
   console.log(weatherObject);
   console.log(weatherText);
   parseWeather(weatherText);       
}
function parseWeather(weatherText)
{
 let weatherJSON = JSON.parse(weatherText);
 console.log(weatherJSON);
 let humidity = weatherJSON.daily[0].humidity;
 let avgTemp = (kToC(weatherJSON.daily[0].temp.min) + kToC(weatherJSON.daily[0].temp.max))/2;
 document.getElementById('humidity').value = humidity;
 document.getElementById('temperature').value = avgTemp;
}
function parseAdd(addText)
{
 let addJSON = JSON.parse(addText);
 console.log(addJSON);
    let data = addJSON["data"];
    for(var i=0; i<Object.keys(data).length; i++)
    {
      if(data[i].county!=null)
      {
        data = data[i];
        break;
      }
      
    }
 let district="";
 console.log(data);
 if(data!="")
 {
  district= data.county;
  console.log(district);
 }
 for(var i=0; i<Object.keys(dist).length; i++)
{
  if(dist[i].District==district)
  {
    // alert(district);
    record=dist[i];
    break;
  }
}
if(record)
{
  console.log("Hello Record");
  if(range)
{
  console.log("Hi!");
  console.log(range);
}
  PhS = record["Ph Stats"];
  NS = record["N Stats"];
  PS = record["P Stats"];
  KS = record["K Stats"];
      if(PhS!="NA")
      {
        document.getElementById("ph").value=range.Ph[PhS].value;
      }
      if(NS!="NA")
      {
        document.getElementById("N").value=range.N[NS].value;
      }
      if(PS!="NA")
      {
        document.getElementById("P").value=range.P[PS].value;
      }
      if(KS!="NA")
      {
        document.getElementById("K").value=range.K[KS].value;
      }
    
}

//  document.getElementById('humidity').innerHTML = humidity;
//  document.getElementById('temperature').innerHTML = avgTemp;
}
let kToC = function(kelvinTemp) {
            const celsius = Math.floor(kelvinTemp - 273);
            // const fahrenheit = Math.floor(celsius * (9 / 5) + 32);
            // return fahrenheit
            return celsius;
        }
function trigger( )
  { 
    console.log("Triggered");
    navigator.geolocation.getCurrentPosition(gotPosition);
    event.preventDefault();
    event.stopPropagation();
  }

// load the data...
let filename = '..\\static\\data\\CropForecast.csv';
d3.csv(filename).then(function(loadedData) {
  
  // the labels will be the same for both
  // (the year, in this case)
  let labels = [];
  
  // ...but we'll make two datasets
  let modal = [];
  let forecast = [];
  
  for (let i=1530; i<loadedData.length; i++) {
    
    // get the year, just like before
    let dStr = loadedData[i].dStr;
    labels.push(dStr);
    let lower;
    let higher;
    // get both data-points, add them to their
    // respective lists
    if(loadedData[i].modal)
    {
     lower =  loadedData[i].modal; 
    }
    else
    {
     lower = Number.NaN;
    }
    if(loadedData[i].forecast)
    {
     higher =  loadedData[i].forecast; 
    }
    else
    {
     higher = Number.NaN;
    }
    // let lower =  loadedData[i].rcp45_weighted_mean;
    // let higher = loadedData[i].rcp85_weighted_mean;
    modal.push(lower);
    forecast.push(higher);
  }
  
  // basic line chart settings
  let options = 
  {
    type: 'line',
    data: {
      labels: labels,   // the labels we loaded!
      
      // add both our datasets here
      datasets: [{
        data: modal,
        label: 'Past Modal Price',
        fill: false,
        // borderColor: 'rgb(255,150,0)'
         borderColor: 'rgb(100,100,100)'
         // borderColor: 'rgb(0,150,255)'
      }, {
        data: forecast,
        label: 'Predicted Modal Price',
        fill: false,
        // borderColor: 'rgb(0,150,255)'
        borderColor: 'rgb(255,150,0)'
      }]},
      options: 
      { plugins: 
        { title: 
          {
                display: true,
                text: 'Apple(₹/Quintal)'
          }
        }
        
      }
  };
  
  // with all that done, we can create our chart!
  let chart = new Chart(document.getElementById('canvas'), options);
});


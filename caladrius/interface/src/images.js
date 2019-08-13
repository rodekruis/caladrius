function importAll(r) {
  let images = {};
  r.keys().map((item, index) => { 
    return(images[item.replace('./', '')] = r(item)) 
  });
  return images;
}

module.exports.images = importAll(
  require.context('./data/Sint-Maarten-2017/test/',  true, /\.png$/));

module.exports.image_placeholder = require('./images/510-logo.png')

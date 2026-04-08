



const calcScaled = (imageWidth, imageHeight, width, height) => {

    const ratioWidth = width / imageWidth;
    const ratioHeight = height / imageHeight;

    const ratio = ratioWidth > ratioHeight ? ratioWidth : ratioHeight

    const newWidth = Math.round(imageWidth * ratio);
    const newHeight = Math.round(imageHeight * ratio);

    console.log(newWidth, newHeight)

    if(newWidth > width){
            
      const sWidth = Math.round(width * (1 / ratio))
      const sx = Math.round((imageWidth - sWidth) / 2)
    
      return {sx:sx, sy:0, sWidth:sWidth, sHeight:imageHeight, dx:0, dy:0, dWidth:width, dHeight:height}
    }
    else if(newHeight > height){
  
      const sHeight = Math.round(height * (1 / ratio))
      const sy = Math.round((imageHeight - sHeight) / 2)
          
      return {sx:0, sy:sy, sWidth:imageWidth, sHeight:sHeight, dx:0, dy:0, dWidth:width, dHeight:height}
    }
    else{

      return {sx:0, sy:sy, sWidth:imageWidth, sHeight:imageHeight, dx:0, dy:0, dWidth:width, dHeight:height}
    }
}


const getBlob = (canvas) => {

  return new Promise((resolve) => {

    canvas.toBlob((blob) => {

      resolve(blob)

      })
  })
}



const imageResize = (url, width, height) => {

    return new Promise((resolve) => {

      const img = new Image();
      //img.src = path;
      
      img.src = url
      img.crossOrigin = 'Anonymous';

      img.onload = async() => {
          
          const scaled = calcScaled(img.width, img.height, width, height)

          const canvas = document.createElement('canvas');
          canvas.width = scaled.dWidth;
          canvas.height = scaled.dHeight;
          const ctx = canvas.getContext('2d');          

          ctx.drawImage(img, scaled.sx, scaled.sy, scaled.sWidth, scaled.sHeight, scaled.dx, scaled.dy, scaled.dWidth, scaled.dHeight);
          
          resolve(await getBlob(canvas))
      }

      img.onerror = () => {

          resolve(null)
      }
    })
}



const test = async() => {

    const url = 'https://upload.wikimedia.org/wikipedia/commons/6/60/%22_Le_Sanitor_%22%2C_d%C3%A9sinfectant_sans_odeur%2C_sant%C3%A9_par_l%27hygi%C3%A8ne_%28...%29%2C_antiseptique%2C_antiputride%2C_anti%C3%A9pid%C3%A9mique._Audibert_%26_Cie%2C_usine_et_administration_68%2C_Boul%28var%29d_St_Marcel%2C_4%2C_rue_Scipion%2C_Paris._En_vente..._-_btv1b9005629d.jpg'
                
    const blob = await imageResize(url, 512, 512)
    
    if(blob == null)
        return null

    const imageUrl = URL.createObjectURL(blob);

    return imageUrl
}

test().then((url) => {

    window.open(url, '_blank');    
})

















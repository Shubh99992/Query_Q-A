const typetext = document.querySelector(".typed-text")
const cursor = document.querySelector(".cursor")
const words = [ "Your Data, Our Expertise." , " Ask Data, Get Answers.","Making Data Work for You."] 
const typingdelay= 100
const erasingdelay = 50 
const newletterdelay = 500
let index = 0;
let charindex = 0
// document.addEventListener("DOMContentLoader" , ()=>{
//     if(words.length){
//         setTimeout(type, newletterdelay)
//     }
// })
document.addEventListener("DOMContentLoaded", () => {
    if (words.length) {
        setTimeout(type, newletterdelay);
    }
});

function type(){
    if ( charindex < words[index].length){
        typetext.textContent +=words[index].charAt(charindex)
        charindex++
        setTimeout(type, typingdelay)
    }else{
        setTimeout(erase, newletterdelay)
    }
}
function erase(){
    if(charindex > 0){
        typetext.textContent = words[index].substring(0, charindex-1)
        charindex--
        setTimeout(erase, erasingdelay)
    }else{
        index++;
        if(index >= words.length){
            index = 0 
        }
        setTimeout(type, typingdelay+100)
    }
}
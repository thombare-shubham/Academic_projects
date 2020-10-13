const express = require('express');
const path = require('path');
const app = express();  //created app with express
const mongoose = require('mongoose');
const bodyparser = require('body-parser');
mongoose.connect('mongodb://localhost/contactDance',{useNewUrlParser:true,useUnifiedTopology: true});
const port = 80; 

//Define mongoose schema
var contactSchema = new mongoose.Schema({
    name : String,
    phone : String,
    email : String,
    address : String,
    desc : String
});

var Contact = mongoose.model('Contact',contactSchema);

//Express Specific Stuff
app.use('/static',express.static('static'));//For serving static files
app.use(express.urlencoded());//helps when we have to take some data from website eg.form data

//Pug Specific stuff
app.set('view engine','pug');//set the template engine as pug
app.set('views',path.join(__dirname,'views'));//set the views directory

//EndPoints
app.get('/',(req,res)=>{
    const params = {};
    res.status(200).render('/views/home.pug',params);
});
app.get('/contact',(req,res)=>{
    const params = {};
    res.status(200).render('/views/contact.pug',params);
});
app.post('/contact',(req,res)=>{
    // const params = {};
    var myData = new Contact(req.body);
    myData.save().then(()=>{
        res.send("This item has been saved to database")
    }).catch(()=>{
        res.status(400).send("Item was not saved to the database");
    })
    // res.status(200).render('contact.pug',params);
});

//Start the server
app.listen(port,()=>{
    console.log(`The application started successfully on port ${port}`);
});
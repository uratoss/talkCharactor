var express = require('express');
var moment = require('moment');
var exec = require('child_process').exec;
var router = express.Router();

router.get('/',function (req,res,next) {
    res.render('index',{});
 });

router.post('/',function (req,res,next) {
    console.log(req.body.xs);

    var com = 'python python/generator_commandline.py '+req.body.xs+' -m predictor.npz';
    exec(com,function (error,stdout,stderr) {
        if(error !== null){
          console.log('exec error: '+error);
          return;
        }
        res.send(stdout);
        console.log(stdout);
    });
});

module.exports = router;

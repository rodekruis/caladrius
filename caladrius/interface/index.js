const express = require("express");
const path = require("path");
const bodyParser = require("body-parser");
const server = require("./server");
const cors = require("cors");
const cookieParser = require("cookie-parser");

const app = express();

const CLIENT_NAME = "*";
const CLIENT_BUILD = "/client/build";

var corsOptions = {
    origin: CLIENT_NAME,
    credentials: true,
    optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));

// Serve the static files from the React app
app.use(express.static(path.join(__dirname, CLIENT_BUILD)));
app.use(express.static("../../data/Sint-Maarten-2017/test"));

app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies

app.use(cookieParser());

app.get("/api/dataset", server.get_dataset_file);

app.get("/api/models", server.get_models);

app.get("/api/model/predictions", server.get_predictions);

// Handles any requests that don't match the ones above
app.get("*", (req, res) => {
    res.sendFile(path.join(__dirname + CLIENT_BUILD + "/index.html"));
});

const port = process.env.PORT || 5000;
app.listen(port, () => {
    console.log("App is listening on port " + port);
});

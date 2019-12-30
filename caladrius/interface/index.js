const express = require("express");
const path = require("path");
const bodyParser = require("body-parser");
const server = require("./server");
const cors = require("cors");
const cookieParser = require("cookie-parser");
const Auth = require("./auth");
const Config = require("./config");

const app = express();

var corsOptions = {
    origin: Config.CLIENT_NAME,
    credentials: true,
    optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));

app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies

app.use(cookieParser());

// open
app.use(express.static(path.join(__dirname, Config.CLIENT_BUILD)));

app.get("/terms-and-conditions", (req, res) => {
    res.sendFile(path.join(__dirname + "/terms_and_conditions.txt"));
});

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname + Config.CLIENT_BUILD + "/index.html"));
});

// authentication
app.use((req, res, next) => {
    req.cookies[Config.COOKIE_NAME] || req.path === "/api/login"
        ? next()
        : res.status(403).json(false);
});

app.get("/api/auth", (req, res) => {
    res.json(req.cookies[Config.COOKIE_NAME]);
});

app.post("/api/login", Auth.login);

app.get("/api/logout", Auth.logout);

// static
app.use(express.static("../../data/Sint-Maarten-2017/validation"));
app.use(express.static("../../data/Sint-Maarten-2017/test"));
app.use(express.static("../../data/Sint-Maarten-2017/inference"));

// backend API
app.get("/api/models", server.get_models);
app.get("/api/datasets", server.get_datasets);

app.get("/api/:model/predictions/:epoch?", server.get_model_predictions);

app.get("/api/:dataset/:filename?", server.get_dataset_file);

// Handles any requests that don't match the ones above
app.get("*", (req, res) => {
    res.sendFile(path.join(__dirname + Config.CLIENT_BUILD + "/index.html"));
});

// listener
const port = process.env.PORT || 5000;
app.listen(port, () => {
    console.log("App is listening on port " + port);
});

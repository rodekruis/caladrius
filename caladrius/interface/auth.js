const fs = require("fs");
const bcrypt = require("bcrypt");
const Config = require("./config");

const authenticate = (username, password, callback) => {
    fs.readFile(Config.CREDENTIALS_LIST, "utf8", async (err, contents) => {
        if (err) callback(false);
        let is_authenticated = false;
        const credentials = contents.split("\n");
        for (let credentialIndex in credentials) {
            const credential = credentials[credentialIndex].split(" ");
            if (username.toLowerCase() === credential[0]) {
                is_authenticated = await bcrypt.compare(
                    password,
                    credential[1]
                );
                break;
            }
        }
        callback(is_authenticated);
    });
};

class Auth {
    login(req, res) {
        const username = req.body.username;
        const password = req.body.password;
        authenticate(username, password, is_authenticated => {
            if (is_authenticated) {
                res.cookie(Config.COOKIE_NAME, username, {
                    httpOnly: true,
                });
            } else {
                res.clearCookie(Config.COOKIE_NAME);
            }
            res.json(is_authenticated);
        });
    }

    logout(req, res) {
        res.clearCookie(Config.COOKIE_NAME);
        res.json(true);
    }

    hash(req, res) {
        const password = req.query.password;
        bcrypt.hash(password, 10, (err, hash) => {
            if (err) res.send(false);
            res.send(hash);
        });
    }
}

module.exports = new Auth();

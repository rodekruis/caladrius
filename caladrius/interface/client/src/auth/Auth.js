export const Auth = {
    auth(callback) {
        fetch("/api/auth", { method: "GET" })
            .then(res => res.json())
            .then(callback);
    },
    login(username, password, callback) {
        fetch("/api/login", {
            method: "POST",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                username: username,
                password: password,
            }),
        })
            .then(res => res.json())
            .then(callback);
    },
    logout(callback) {
        fetch("/api/logout", { method: "GET" })
            .then(res => res.json())
            .then(callback);
    },
};

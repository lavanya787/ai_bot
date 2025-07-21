document.addEventListener('DOMContentLoaded', function () {
    // Show/Hide Containers
    window.showLogin = function () {
        document.getElementById('login-container').style.display = 'flex';
        document.getElementById('register-container').style.display = 'none';
    };

    window.showRegister = function () {
        document.getElementById('login-container').style.display = 'none';
        document.getElementById('register-container').style.display = 'flex';
    };

    // Notification Function
    function showNotification(message, duration = 3000) {
        const notification = document.getElementById('notification');
        const notificationMessage = document.getElementById('notification-message');
        notificationMessage.textContent = message;
        notification.style.display = 'block';
        setTimeout(() => {
            notification.style.display = 'none';
        }, duration);
    }

    // Sign-Up Handler
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const username = document.getElementById('register-username').value.trim();
            const email = document.getElementById('register-email').value.trim();
            const password = document.getElementById('register-password').value.trim();

            if (username.length < 3) {
                showNotification('Username must be at least 3 characters long!');
                return;
            }

            const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailPattern.test(email)) {
                showNotification('Please enter a valid email address!');
                return;
            }

            if (password.length < 8) {
                showNotification('Password must be at least 8 characters long!');
                return;
            }

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, email, password })
                });
                const result = await response.json();

                if (response.ok) {
                    showNotification(result.message || 'Registered successfully!');
                    setTimeout(() => {
                        window.location.href = '/'; // Back to login
                    }, 2000);
                } else {
                    showNotification(result.message || 'Registration failed.');
                }
            } catch (error) {
                showNotification('Error during registration.');
                console.error(error);
            }
        });
    }

    // Sign-In Handler
    const signinForm = document.getElementById('signinForm');
    if (signinForm) {
        signinForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const email = document.getElementById('login-email').value.trim();
            const password = document.getElementById('login-password').value.trim();

            const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailPattern.test(email)) {
                showNotification('Please enter a valid email address!');
                return;
            }

            if (password.length < 1) {
                showNotification('Password cannot be empty!');
                return;
            }

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });

                const result = await response.json();

                if (response.ok) {
                    showNotification(result.message || 'Login successful!');
                    setTimeout(() => {
                        window.location.href = result.redirect || '/train'; // ✅ Redirect to train
                    }, 2000);
                } else {
                    showNotification(result.message || 'Login failed.');
                }
            } catch (error) {
                showNotification('Error during login.');
                console.error(error);
            }
        });
    }
});
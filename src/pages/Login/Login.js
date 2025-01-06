import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; 
import styles from './Login.module.css'; 
import UserApi from '../../api/UserApi';
import Cookies from 'js-cookie';
import { useUser } from '../../contexts/UserContext'; 
import { jwtDecode } from 'jwt-decode'; 
import { useEffect } from 'react';

const Login = () => {
    const { login } = useUser();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate(); 

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!email || !password) {
            setError('Please fill in both email and password.');
            return;
        }

        try {
            const response = await UserApi.login(email, password);
            const data = response.data.result;
            console.log(data);

            if (data.authenticated) {
                localStorage.setItem('accessToken', data.accessToken);
                Cookies.set('refreshToken', data.refreshToken, {
                    secure: false,
                });
                const decodedToken = jwtDecode(data.accessToken);
                console.log("DECODED:", decodedToken)
                const userData = {
                    userId: decodedToken.userId,
                    email: decodedToken.sub, 
                    role: decodedToken.role
                };

                login(userData, data.accessToken);

                const isAdmin = decodedToken.role === 'ADMIN';
                if (isAdmin) {
                    navigate('/admin');  
                } else {
                    navigate('/'); 
                }

            } else {
                setError(data.message || 'Invalid login credentials.');
            }
        } catch (error) {
            setError('Error connecting to the server. Please try again later.');
        }
    };

   
    return (
        <div className={styles.loginContainer}> 
            <h2>Login</h2>
            {error && <p className={styles.error}>{error}</p>} 
            <form onSubmit={handleSubmit}>
                <div className={styles.formGroup}> 
                    <label htmlFor="email">Email</label>
                    <input
                        type="email"
                        id="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="Enter your email"
                    />
                </div>

                <div className={styles.formGroup}> 
                    <label htmlFor="password">Password</label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter your password"
                    />
                </div>

                <button type="submit">Login</button>
            </form>
        </div>
    );
};

export default Login;

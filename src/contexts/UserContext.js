import React, { createContext, useState, useEffect, useContext } from 'react';
import { jwtDecode } from 'jwt-decode'; 
import  axiosInstance from '../api/UserApi';
import Cookies from 'js-cookie';

const UserContext = createContext();

export const useUser = () => {
    return useContext(UserContext);
};

export const UserProvider = ({ children }) => {
    const [user, setUser] = useState({}); 
    const [token, setToken] = useState(localStorage.getItem('accessToken') || ''); 
    const [isRefreshing, setIsRefreshing] = useState(false); 

    useEffect(() => {
        if (token) {
            try {
                const decodedToken = jwtDecode(token);

                if (decodedToken.exp * 1000 < Date.now()) {
                    console.warn('Token has expired');
                    refreshAccessToken(); 
                    return;
                }

                const userData = {
                    userId: decodedToken.userId,
                    email: decodedToken.sub,
                };
                setUser(userData);
            } catch (error) {
                
                console.error('Invalid token', error);
                logout(); 
            }
        }
    }, [token]);

    const refreshAccessToken = async () => {
        if (isRefreshing) return; 

        setIsRefreshing(true); 
        try {
            const response = await axiosInstance.post('/auth/refresh'); 

            const { accessToken } = response.data.result.accessToken; 

            localStorage.setItem('accessToken', accessToken);
            setToken(accessToken);

            const decodedToken = jwtDecode(accessToken);
            const userData = {
                userId: decodedToken.userId,
                email: decodedToken.sub,
            };
            setUser(userData);
        } catch (error) {
            console.error('Failed to refresh token', error);
            logout(); 
        } finally {
            setIsRefreshing(false); 
        }
    };


    const login = (userData, token) => {
        setUser(userData);
        setToken(token);
        localStorage.setItem('accessToken', token); 
    };

    const logout = () => {
        setUser({}); 
        setToken('');
        localStorage.removeItem('accessToken'); 
        Cookies.remove('refreshToken');
    };

    return (
        <UserContext.Provider value={{ user, token, login, logout }}>
            {children}
        </UserContext.Provider>
    );
};

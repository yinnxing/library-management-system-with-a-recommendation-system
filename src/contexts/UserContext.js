import React, { createContext, useState, useEffect, useContext } from 'react';
import { jwtDecode } from 'jwt-decode'; 
import axiosInstance from '../api/UserApi';
import Cookies from 'js-cookie';

const UserContext = createContext();

export const useUser = () => {
    return useContext(UserContext);
};

export const UserProvider = ({ children }) => {
    const [user, setUser] = useState(null); 
    const [token, setToken] = useState(localStorage.getItem('accessToken') || ''); 
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    // Check authentication on mount and token change
    useEffect(() => {
        const checkAuth = async () => {
            const storedToken = localStorage.getItem('accessToken');
            
            if (!storedToken) {
                setIsAuthenticated(false);
                setUser(null);
                return;
            }
            
            try {
                const decodedToken = jwtDecode(storedToken);

                // Check if token is expired
                if (decodedToken.exp * 1000 < Date.now()) {
                    console.warn('Token has expired');
                    try {
                        await refreshAccessToken();
                    } catch (error) {
                        logout();
                    }
                    return;
                }

                // Set user data from token
                const userData = {
                    userId: decodedToken.userId,
                    email: decodedToken.sub,
                    role: decodedToken.role || 'USER'
                };
                
                setUser(userData);
                setToken(storedToken);
                setIsAuthenticated(true);
            } catch (error) {
                console.error('Invalid token', error);
                logout();
            }
        };

        checkAuth();
    }, [token]);

    const refreshAccessToken = async () => {
        if (isRefreshing) return; 

        setIsRefreshing(true); 
        try {
            const response = await axiosInstance.post('/auth/refresh'); 
            const { accessToken } = response.data.result;

            localStorage.setItem('accessToken', accessToken);
            setToken(accessToken);

            const decodedToken = jwtDecode(accessToken);
            const userData = {
                userId: decodedToken.userId,
                email: decodedToken.sub,
                role: decodedToken.role || 'USER'
            };
            
            setUser(userData);
            setIsAuthenticated(true);
            return accessToken;
        } catch (error) {
            console.error('Failed to refresh token', error);
            logout();
            throw error;
        } finally {
            setIsRefreshing(false); 
        }
    };

    const login = (userData, newToken) => {
        setUser(userData);
        setToken(newToken);
        setIsAuthenticated(true);
        localStorage.setItem('accessToken', newToken); 
    };

    const logout = () => {
        setUser(null); 
        setToken('');
        setIsAuthenticated(false);
        localStorage.removeItem('accessToken'); 
        Cookies.remove('refreshToken');
    };

    return (
        <UserContext.Provider value={{ 
            user, 
            token, 
            login, 
            logout, 
            isAuthenticated,
            refreshToken: refreshAccessToken
        }}>
            {children}
        </UserContext.Provider>
    );
};

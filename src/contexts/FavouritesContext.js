import React, { createContext, useState, useContext } from 'react';

const FavoritesContext = createContext();

export const FavoritesProvider = ({ children }) => {
    const [favoriteBooks, setFavoriteBooks] = useState([]);

    const addToFavorites = (bookId) => {
        setFavoriteBooks((prevFavorites) => [...prevFavorites, bookId]);
    };

    const removeFromFavorites = (bookId) => {
        setFavoriteBooks((prevFavorites) => prevFavorites.filter((id) => id !== bookId));
    };

    return (
        <FavoritesContext.Provider value={{ favoriteBooks, addToFavorites, removeFromFavorites }}>
            {children}
        </FavoritesContext.Provider>
    );
};

export const useFavorites = () => {
    return useContext(FavoritesContext);
};

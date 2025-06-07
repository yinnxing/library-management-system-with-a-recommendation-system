import React from "react";
import {useState, useEffect } from 'react';
import styles from "./BookDetails.module.css";
import { useParams, useNavigate } from "react-router-dom";
import books from '../../assets/books'
import UserApi from '../../api/UserApi';
import { useUser } from '../../contexts/UserContext';

const BookDetails = () => {
    const { bookId } = useParams(); 
    const navigate = useNavigate();
    const { user } = useUser();
    const [book, setBook] = useState(null); 
    const [loading, setLoading] = useState(true); 
    const [error, setError] = useState(null); 
    
    // Review states
    const [reviews, setReviews] = useState([]);
    const [reviewsLoading, setReviewsLoading] = useState(false);
    const [newReview, setNewReview] = useState({
        rating: 0,
        comment: ''
    });
    const [submittingReview, setSubmittingReview] = useState(false);
    const [reviewError, setReviewError] = useState(null);
    const [reviewSuccess, setReviewSuccess] = useState(false);

    useEffect(() => {
        // Scroll to top when component loads
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        const fetchBookDetails = async () => {
            try {
                setLoading(true); 
                const response = await UserApi.getBook(bookId); 
                const data = response.data.result;
                if (!data) {
                    throw new Error('Failed to fetch book details');
                }
                setBook(data); 
            } catch (error) {
                setError(error.message); 
            } finally {
                setLoading(false); 
            }
        };

        fetchBookDetails();
        fetchReviews();
    }, [bookId]); 

    const fetchReviews = async () => {
        try {
            setReviewsLoading(true);
            const response = await UserApi.getBookReviews(bookId);
            if (response.data.code === 0) {
                // Handle paginated response structure
                const reviewsData = response.data.result.content || [];
                setReviews(reviewsData);
            }
        } catch (error) {
            console.error('Error fetching reviews:', error);
        } finally {
            setReviewsLoading(false);
        }
    };

    const handleBorrowClick = () => {
        navigate(`/borrow/${bookId}`);
    };

    const handleBackClick = () => {
        navigate(-1); // Go back to previous page
    };

    const handleRatingClick = (rating) => {
        setNewReview(prev => ({ ...prev, rating }));
    };

    const handleCommentChange = (e) => {
        setNewReview(prev => ({ ...prev, comment: e.target.value }));
    };

    const handleSubmitReview = async (e) => {
        e.preventDefault();
        
        if (!user) {
            setReviewError('Vui lòng đăng nhập để đánh giá sách.');
            return;
        }

        if (newReview.rating === 0) {
            setReviewError('Vui lòng chọn số sao đánh giá.');
            return;
        }

        if (!newReview.comment.trim()) {
            setReviewError('Vui lòng nhập nhận xét.');
            return;
        }

        try {
            setSubmittingReview(true);
            setReviewError(null);

            const reviewData = {
                userId: user.userId,
                bookId: parseInt(bookId),
                rating: newReview.rating,
                comment: newReview.comment.trim()
            };

            const response = await UserApi.submitReview(reviewData);

            if (response.data.code === 0) {
                setReviewSuccess(true);
                setNewReview({ rating: 0, comment: '' });
                
                // Add the new review to the list
                setReviews(prev => [response.data.result, ...prev]);
                
                // Clear success message after 3 seconds
                setTimeout(() => setReviewSuccess(false), 3000);
            } else {
                setReviewError(response.data.message || 'Có lỗi xảy ra khi gửi đánh giá.');
            }
        } catch (error) {
            console.error('Error submitting review:', error);
            const errorMessage = error.response?.data?.message || 'Không thể gửi đánh giá. Vui lòng thử lại sau.';
            setReviewError(errorMessage);
        } finally {
            setSubmittingReview(false);
        }
    };

    const renderStars = (rating, interactive = false, size = 'medium') => {
        const stars = [];
        for (let i = 1; i <= 5; i++) {
            stars.push(
                <span
                    key={i}
                    className={`${styles.star} ${i <= rating ? styles.filled : styles.empty} ${interactive ? styles.interactive : ''} ${styles[size]}`}
                    onClick={interactive ? () => handleRatingClick(i) : undefined}
                >
                    ⭐
                </span>
            );
        }
        return stars;
    };

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleDateString('vi-VN', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const calculateAverageRating = () => {
        if (reviews.length === 0) return 0;
        const sum = reviews.reduce((acc, review) => acc + review.rating, 0);
        return (sum / reviews.length).toFixed(1);
    };

    if (loading) {
        return (
            <div className={styles.container}>
                <div className={styles.loading}>
                    <div className={styles.loadingSpinner}></div>
                    <p className={styles.loadingText}>Loading book details...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.container}>
                <div className={styles.error}>
                    <div className={styles.errorIcon}>⚠️</div>
                    <p className={styles.errorText}>Error: {error}</p>
                </div>
            </div>
        );
    }

    const {
        title,
        author,
        publisher,
        publicationYear,
        isbn,
        genre,
        descriptions,
        coverImageUrl,
        quantity,
        availableQuantity,
        previewLink,
    } = book || {}; 

    // Helper function to create Google Books search URL
    const getGoogleBooksSearchUrl = () => {
        const searchQuery = encodeURIComponent(`${title} ${author}`);
        return `https://www.google.com/search?q=${searchQuery}+site:books.google.com`;
    };

    // Helper function to create Amazon search URL
    const getAmazonSearchUrl = () => {
        const searchQuery = encodeURIComponent(`${title} ${author}`);
        return `https://www.amazon.com/s?k=${searchQuery}&i=stripbooks`;
    };

    return (
        <div className={styles.container}>
            {/* Navigation Header */}
            <div className={styles.navigationHeader}>
                <button onClick={handleBackClick} className={styles.backButton}>
                    ← Quay lại
                </button>
                <div className={styles.pageActions}>
                    <button 
                        onClick={handleBorrowClick}
                        disabled={availableQuantity === 0}
                        className={`${styles.borrowActionButton} ${availableQuantity === 0 ? styles.disabled : ''}`}
                    >
                        {availableQuantity > 0 ? (
                            <>
                                <span className={styles.buttonIcon}>📖</span>
                                Mượn sách
                            </>
                        ) : (
                            <>
                                <span className={styles.buttonIcon}>❌</span>
                                Hết sách
                            </>
                        )}
                    </button>
                </div>
            </div>

            <div className={styles.bookDetailsCard}>
                {/* Header Section with Cover and Main Info */}
                <div className={styles.headerSection}>
                    <div className={styles.coverImageContainer}>
            <img
                src={coverImageUrl}
                alt={`${title} cover`}
                className={styles.coverImage}
            />
                        {/* <div className={styles.imageOverlay}>
                            High Quality Cover
                        </div> */}
                    </div>
                    
                    <div className={styles.bookInfoSection}>
                        <div className={styles.titleSection}>
                            <h1 className={styles.bookTitle}>{title}</h1>
                            <p className={styles.bookSubtitle}>by {author}</p>
                            
                            {/* Rating Summary */}
                            {reviews.length > 0 && (
                                <div className={styles.ratingSummary}>
                                    <div className={styles.averageRating}>
                                        <span className={styles.ratingNumber}>{calculateAverageRating()}</span>
                                        <div className={styles.ratingStars}>
                                            {renderStars(Math.round(calculateAverageRating()), false, 'small')}
                                        </div>
                                        <span className={styles.reviewCount}>({reviews.length} đánh giá)</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Metadata Grid */}
                        <div className={styles.metadataGrid}>
                            <div className={styles.metadataItem}>
                                <div className={styles.metadataIcon}>📖</div>
                                <div className={styles.metadataContent}>
                                    <p className={styles.metadataLabel}>Publisher</p>
                                    <p className={styles.metadataValue}>{publisher}</p>
                                </div>
                            </div>

                            <div className={styles.metadataItem}>
                                <div className={styles.metadataIcon}>📅</div>
                                <div className={styles.metadataContent}>
                                    <p className={styles.metadataLabel}>Publication Year</p>
                                    <p className={styles.metadataValue}>{publicationYear}</p>
                                </div>
                            </div>

                            <div className={styles.metadataItem}>
                                <div className={styles.metadataIcon}>🏷️</div>
                                <div className={styles.metadataContent}>
                                    <p className={styles.metadataLabel}>ISBN</p>
                                    <p className={styles.metadataValue}>{isbn}</p>
                                </div>
                            </div>

                            <div className={styles.metadataItem}>
                                <div className={styles.metadataIcon}>📚</div>
                                <div className={styles.metadataContent}>
                                    <p className={styles.metadataLabel}>Genre</p>
                                    <p className={styles.metadataValue}>{genre || "General"}</p>
                                </div>
                            </div>
                        </div>

                        {/* Availability Section */}
                        <div className={styles.availabilitySection}>
                            <div className={styles.availabilityItem}>
                                <p className={styles.availabilityNumber}>{quantity}</p>
                                <p className={styles.availabilityLabel}>Total Copies</p>
                            </div>
                            <div className={styles.availabilityItem}>
                                <p className={styles.availabilityNumber}>{availableQuantity}</p>
                                <p className={styles.availabilityLabel}>Available</p>
                            </div>
                            
                        </div>

                        {/* Borrow Button Section */}
                        <div className={styles.borrowSection}>
                            <button 
                                onClick={handleBorrowClick}
                                disabled={availableQuantity === 0}
                                className={styles.borrowButton}
                            >
                                {availableQuantity > 0 ? (
                                    <>
                                        <span className={styles.borrowIcon}>📖</span>
                                        Mượn sách ngay
                                        <span className={styles.borrowArrow}>→</span>
                                    </>
                                ) : (
                                    <>
                                        <span className={styles.borrowIcon}>❌</span>
                                        Sách đã hết
                                    </>
                                )}
                            </button>
                            {availableQuantity > 0 && (
                                <p className={styles.borrowNote}>
                                    Nhấn để chuyển đến trang mượn sách với đầy đủ thông tin và chính sách
                                </p>
                            )}
                        </div>
                    </div>
                </div>

                {/* Description Section */}
                <div className={styles.descriptionSection}>
                    <h2 className={styles.sectionTitle}>Description</h2>
                    <p className={styles.description}>
                        {descriptions || "No description available for this book. This is a wonderful addition to our library collection that offers valuable insights and knowledge to our readers."}
                    </p>
                </div>

                {/* Reviews Section */}
                <div className={styles.reviewsSection}>
                    <h2 className={styles.sectionTitle}>Đánh giá và nhận xét</h2>
                    
                    {/* Add Review Form */}
                    {user ? (
                        <div className={styles.addReviewSection}>
                            <h3 className={styles.addReviewTitle}>
                                <span className={styles.reviewIcon}>✍️</span>
                                Viết đánh giá của bạn
                            </h3>
                            
                            <form onSubmit={handleSubmitReview} className={styles.reviewForm}>
                                <div className={styles.ratingInput}>
                                    <label className={styles.ratingLabel}>Đánh giá:</label>
                                    <div className={styles.starRating}>
                                        {renderStars(newReview.rating, true, 'large')}
                                    </div>
                                </div>
                                
                                <div className={styles.commentInput}>
                                    <label className={styles.commentLabel}>Nhận xét:</label>
                                    <textarea
                                        value={newReview.comment}
                                        onChange={handleCommentChange}
                                        placeholder="Chia sẻ cảm nhận của bạn về cuốn sách này..."
                                        className={styles.commentTextarea}
                                        rows={4}
                                    />
                                </div>
                                
                                {reviewError && (
                                    <div className={styles.reviewError}>
                                        <span className={styles.errorIcon}>⚠️</span>
                                        {reviewError}
                                    </div>
                                )}
                                
                                {reviewSuccess && (
                                    <div className={styles.reviewSuccess}>
                                        <span className={styles.successIcon}>✅</span>
                                        Đánh giá của bạn đã được gửi thành công!
                                    </div>
                                )}
                                
                                <button
                                    type="submit"
                                    disabled={submittingReview}
                                    className={styles.submitReviewButton}
                                >
                                    {submittingReview ? (
                                        <>
                                            <div className={styles.buttonSpinner}></div>
                                            Đang gửi...
                                        </>
                                    ) : (
                                        <>
                                            <span className={styles.submitIcon}>📝</span>
                                            Gửi đánh giá
                                        </>
                                    )}
                                </button>
                            </form>
                        </div>
                    ) : (
                        <div className={styles.loginPrompt}>
                            <span className={styles.loginIcon}>🔐</span>
                            <p>Vui lòng đăng nhập để viết đánh giá</p>
                        </div>
                    )}
                    
                    {/* Reviews List */}
                    <div className={styles.reviewsList}>
                        <h3 className={styles.reviewsListTitle}>
                            <span className={styles.reviewsIcon}>💬</span>
                            Đánh giá từ độc giả ({reviews.length})
                        </h3>
                        
                        {reviewsLoading ? (
                            <div className={styles.reviewsLoading}>
                                <div className={styles.loadingSpinner}></div>
                                <p>Đang tải đánh giá...</p>
                            </div>
                        ) : reviews.length > 0 ? (
                            <div className={styles.reviewsContainer}>
                                {reviews.map((review) => (
                                    <div key={review.reviewId} className={styles.reviewItem}>
                                        <div className={styles.reviewHeader}>
                                            <div className={styles.reviewerInfo}>
                                                <span className={styles.reviewerAvatar}>👤</span>
                                                <div className={styles.reviewerDetails}>
                                                    <span className={styles.reviewerName}>{review.username}</span>
                                                    <span className={styles.reviewDate}>{formatDate(review.createdAt)}</span>
                                                </div>
                                            </div>
                                            <div className={styles.reviewRating}>
                                                {renderStars(review.rating, false, 'small')}
                                            </div>
                                        </div>
                                        <div className={styles.reviewContent}>
                                            <p className={styles.reviewComment}>{review.comment}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className={styles.noReviews}>
                                <span className={styles.noReviewsIcon}>📝</span>
                                <p>Chưa có đánh giá nào cho cuốn sách này. Hãy là người đầu tiên!</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Preview Section */}
                <div className={styles.previewSection}>
                    <h2 className={styles.sectionTitle}>Book Preview</h2>
                    <div className={styles.previewContainer}>
                        <div className={styles.previewUnavailable}>
                            <h3>📖 Preview Not Available</h3>
                            <p>
                                The book preview cannot be displayed directly due to access restrictions. 
                                However, you can explore this book through the following options:
                            </p>
                            <div className={styles.previewAlternatives}>
                                <a 
                                    href={getGoogleBooksSearchUrl()} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className={styles.previewButton}
                                >
                                    🔍 Search on Google Books
                                </a>
                                <a 
                                    href={getAmazonSearchUrl()} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className={`${styles.previewButton} ${styles.secondary}`}
                                >
                                    🛒 Find on Amazon
                                </a>
                                {previewLink && (
                                    <a 
                                        href={previewLink} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className={styles.previewButton}
                                    >
                                        📱 Try Original Link
                                    </a>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BookDetails;
